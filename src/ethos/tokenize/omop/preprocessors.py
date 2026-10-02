import numpy as np
import polars as pl

from ...constants import SpecialToken as ST
from ..patterns import MatchAndRevise
from ..utils import apply_vocab_to_multitoken_codes, unify_code_names

admission_codes = [ST.ADMISSION, "Visit/IP", "Visit/ERIP", "CMS Place of Service/51",
                   "CMS Place of Service/61"]
discharge_codes = ["CMS Place of Service", "SNOMED/371827001", "SNOMED/397709008",
                   "SNOMED/225928004", "Medicare Specialty/A4", "PCORNet/Generic-"]


class DeathData:
    @staticmethod
    @MatchAndRevise(prefix=[ST.DEATH] + discharge_codes, needs_resorting=True)
    def place_death_before_dc_if_same_time(df: pl.DataFrame) -> pl.DataFrame:
        gb_cols = MatchAndRevise.sort_cols
        idx_col = MatchAndRevise.index_col
        return (
            df.sort(pl.col("code").replace_strict(ST.DEATH, 0, default=1, return_dtype=pl.UInt8))
            .group_by(gb_cols, maintain_order=True)
            .agg(pl.col(idx_col).last(), pl.exclude(gb_cols, idx_col))
            .explode(pl.exclude(gb_cols, idx_col))
            .sort(by=idx_col)
            .select(df.columns)
        )


class DemographicData:
    @staticmethod
    @MatchAndRevise(prefix=admission_codes)
    def retrieve_demographics_from_hosp_adm(df: pl.DataFrame) -> pl.DataFrame:
        return df

    @staticmethod
    @MatchAndRevise(prefix="Gender/")
    def unify_gender_code(df: pl.DataFrame) -> pl.DataFrame:
        """meds_etl.omop emits 'Gender/F' style codes; StaticDataCollector expects the
        double-slash 'GENDER//F' convention used elsewhere in the pipeline."""
        return df.with_columns(
            code=pl.lit("GENDER//") + pl.col("code").str.split_exact("/", 1).struct[1]
        )


class InpatientData:
    @staticmethod
    @MatchAndRevise(prefix=[ST.ADMISSION, "Visit/IP", "Visit/ERIP", "CMS Place of Service/51",
                            "CMS Place of Service/61"])
    def process_hospital_admissions(df: pl.DataFrame) -> pl.DataFrame:
        return df

    @staticmethod
    @MatchAndRevise(prefix=[ST.DISCHARGE, "ICD10CM//", "ICD9CM//", "DRG//"])
    def process_hospital_discharges(df: pl.DataFrame) -> pl.DataFrame:
        """Currently must be run before processing diagnoses."""
        discharge_facilities = [
            "HEALTHCARE FACILITY",
            "SKILLED NURSING FACILITY",
            "REHAB",
            "CHRONIC/LONG TERM ACUTE CARE",
            "OTHER FACILITY",
        ]

        drg_following_diag = pl.col.code.str.starts_with(
            "DIAGNOSIS//ICD"
        ) & ~pl.col.code.str.starts_with("DRG//").shift(-1, fill_value=False)
        drg_following_disch = pl.col.code.str.starts_with(ST.DISCHARGE)

        drg_following_diag &= ~pl.col.code.str.starts_with("DIAGNOSIS//ICD").shift(
            -1, fill_value=False
        )
        drg_following_disch &= pl.col.code.str.starts_with(ST.DISCHARGE).shift(
            -1, fill_value=True
        )

        drg_missing_cond = drg_following_diag | drg_following_disch

        return (
            df.with_columns(
                text_value=pl.when(pl.col.code.str.starts_with(ST.DISCHARGE))
                .then(pl.col.code.str.split("//").list[1])
                .otherwise("text_value")
            )
            .with_columns(
                code=pl.when(pl.col.code.str.starts_with(ST.DISCHARGE))
                .then(
                    pl.concat_list(
                        pl.lit(ST.DISCHARGE),
                        (
                            pl.lit("DISCHARGE_LOCATION//")
                            + pl.when(pl.col("text_value").is_in(discharge_facilities))
                            .then(pl.lit("HEALTHCARE_FACILITY"))
                            .when(pl.col("text_value").is_null())
                            .then(pl.lit("UNKNOWN"))
                            .otherwise(pl.col("text_value").replace(" ", "_"))
                        ),
                    )
                )
                .otherwise(pl.concat_list("code")),
                drg_missing=drg_missing_cond,
            )
            .with_columns(
                code=pl.when("drg_missing")
                .then(pl.concat_list("code", pl.lit("DRG//UNKNOWN")))
                .otherwise("code")
            )
            .drop("drg_missing")
            .explode("code")
        )


class MeasurementData:
    @staticmethod
    @MatchAndRevise(prefix=["LOINC"])
    def process_simple_measurements(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.filter(pl.col("numeric_value").is_not_null())
            .with_columns(
                code=pl.concat_list(
                    pl.lit("LOINC//") + pl.col("code"), pl.lit("LOINC//Q//") + pl.col("code")
                )
            )
            .explode("code")
        )


def _normalize_icd_codes(codes: pl.Series, known_codes: set[str]) -> dict[str, str]:
    """Removes dots from ICD codes and, until a code is known, strips trailing zeros that some
    OMOP sources use to pad codes to a fixed length (e.g., ICD10CM/I10.00 -> I10)."""
    mapping = {}
    for raw_code in codes.drop_nulls().unique():
        code = raw_code.replace(".", "").upper()
        while code not in known_codes and code.endswith("0") and len(code) > 3:
            code = code[:-1]
        mapping[raw_code] = code
    return mapping


def _convert_icd9_cm_codes(codes: pl.Series, icd_9_to_10: dict[str, str]) -> dict[str, str]:
    """Maps ICD-9-CM codes to ICD-10-CM. Codes without an exact match (e.g., categories such as
    585) are mapped to the ICD-10-CM category of the first matching subcode, which follows how
    ambiguous codes are handled in `get_icd_9_to_10_mapping`."""
    mapping = {}
    for raw_code, code in _normalize_icd_codes(codes, set(icd_9_to_10)).items():
        if code in icd_9_to_10:
            mapping[raw_code] = icd_9_to_10[code]
        elif subcodes := sorted(v for k, v in icd_9_to_10.items() if k.startswith(code)):
            mapping[raw_code] = subcodes[0][:3]
    return mapping


def _split_vocabulary_and_code(df: pl.DataFrame) -> pl.DataFrame:
    vocab_and_code = pl.col("code").str.split_exact("/", 1)
    return df.with_columns(
        vocabulary=vocab_and_code.struct[0], concept_code=vocab_and_code.struct[1]
    )


class DiagnosesData:
    @staticmethod
    @MatchAndRevise(prefix=["ICD10CM/", "ICD9CM/"], needs_vocab=True)
    def split_icd_codes(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        """Follows the MIMIC tokenization: ICD-9-CM codes are converted to ICD-10-CM, and each
        code is split into the category (named after its description), characters 4-6, and the
        suffix, e.g., ICD10CM/I25.10 -> [ICD//CM//<I25 name>, ICD//CM//3-6//10]."""
        from ..mappings import get_icd_cm_9_to_10_mapping, get_icd_cm_code_to_name_mapping

        code_to_name = get_icd_cm_code_to_name_mapping()
        icd_9_to_10 = get_icd_cm_9_to_10_mapping()

        df = _split_vocabulary_and_code(df)
        is_icd9 = pl.col("vocabulary") == "ICD9CM"
        icd9_to_10 = _convert_icd9_cm_codes(df.filter(is_icd9)["concept_code"], icd_9_to_10)
        icd10_norm = _normalize_icd_codes(df.filter(~is_icd9)["concept_code"], set(code_to_name))

        temp_cols = ["part1", "part2", "part3"]
        code_prefixes = ["", "3-6//", "SFX//"]
        df = (
            df.with_columns(
                icd=pl.when(is_icd9)
                .then(
                    pl.col("concept_code").replace_strict(
                        icd9_to_10, default=None, return_dtype=pl.String
                    )
                )
                .otherwise(
                    pl.col("concept_code").replace_strict(
                        icd10_norm, default=None, return_dtype=pl.String
                    )
                )
            )
            .with_columns(
                part1=pl.col("icd")
                .str.slice(0, 3)
                .replace_strict(code_to_name, default=None, return_dtype=pl.String),
                part2=pl.col("icd").str.slice(3, 3),
                part3=pl.col("icd").str.slice(6),
            )
            .with_columns(
                # codes with an unknown category are dropped altogether
                pl.when(pl.col("part1").is_not_null() & (pl.col(col) != ""))
                .then(pl.lit(f"ICD//CM//{prefix}") + pl.col(col))
                .alias(col)
                for col, prefix in zip(temp_cols, code_prefixes)
            )
            .with_columns(unify_code_names(pl.col(temp_cols)))
        )

        if vocab is not None:
            df = apply_vocab_to_multitoken_codes(df, temp_cols, vocab)

        return (
            df.with_columns(code=pl.concat_list(temp_cols))
            .drop(*temp_cols, "vocabulary", "concept_code", "icd")
            .explode("code")
            .drop_nulls("code")
        )


class ProcedureData:
    @staticmethod
    @MatchAndRevise(prefix=["ICD10PCS/", "ICD9Proc/"], needs_vocab=True)
    def split_icd_codes(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        """Follows the MIMIC tokenization: ICD-9 procedure codes are converted to ICD-10-PCS, and
        each code is split into its seven characters, e.g., ICD10PCS/0DTJ4ZZ -> [ICD//PCS//0,
        ICD//PCS//D, ICD//PCS//T, ICD//PCS//J, ICD//PCS//4, ICD//PCS//Z, ICD//PCS//Z]."""
        from ..mappings import get_icd_pcs_9_to_10_mapping

        icd_9_to_10 = get_icd_pcs_9_to_10_mapping()

        df = _split_vocabulary_and_code(df)
        is_icd9 = pl.col("vocabulary") == "ICD9Proc"
        icd9_norm = _normalize_icd_codes(df.filter(is_icd9)["concept_code"], set(icd_9_to_10))

        df = (
            df.with_columns(
                icd=pl.when(is_icd9)
                .then(
                    pl.col("concept_code")
                    .replace_strict(icd9_norm, default=None, return_dtype=pl.String)
                    .replace_strict(icd_9_to_10, default=None, return_dtype=pl.String)
                )
                .otherwise(pl.col("concept_code").str.to_uppercase())
            )
            # drops placeholders such as ICD10PCS/NoPCS
            .filter(pl.col("icd").str.contains(r"^[0-9A-Z]{7}$"))
            .with_columns(
                code=pl.concat_list(
                    pl.lit("ICD//PCS//") + pl.col("icd").str.slice(i, 1) for i in range(7)
                )
            )
        )

        if vocab is not None:
            # all characters have to be in the vocab to keep the code
            df = df.filter(pl.col("code").list.eval(pl.element().is_in(vocab)).list.all())

        return df.drop("vocabulary", "concept_code", "icd").explode("code")


class MedicationData:
    @staticmethod
    @MatchAndRevise(prefix=["RxNorm/", "RxNorm Extension/", "NDC/"], needs_vocab=True)
    def convert_to_atc(
        df: pl.DataFrame, vocab: list[str] | None = None, atc_mapping_fp: str | None = None
    ) -> pl.DataFrame:
        """Follows the MIMIC tokenization: drugs are mapped to their ATC codes, and each ATC code
        is split into the first three characters (with their description), the fourth character,
        and the suffix, e.g., N02BE01 -> [ATC//N02//ANALGESICS, ATC//4//B, ATC//SFX//E01]. Drugs
        without an ATC mapping are dropped.

        The mapping is built from the OMOP vocabulary with `scripts/omop/build_atc_mapping.py`.
        """
        from ..mappings import get_atc_code_to_desc, get_omop_drug_to_atc_mapping

        if atc_mapping_fp is None:
            raise ValueError(
                "`atc_mapping_fp` is required to convert drugs to ATC codes, build it with "
                "`scripts/omop/build_atc_mapping.py`."
            )

        drug_to_atc = get_omop_drug_to_atc_mapping(atc_mapping_fp)
        code_to_desc = get_atc_code_to_desc()
        temp_cols = ["pfx", "4", "sfx"]
        code_prefixes = ["ATC//", "ATC//4//", "ATC//SFX//"]
        code_slices = [(0, 3), (3, 1), (4,)]

        df = (
            df.with_columns(
                atc=pl.col("code").replace_strict(
                    drug_to_atc, default=None, return_dtype=pl.List(pl.String)
                )
            )
            .drop_nulls("atc")
            .explode("atc")
            .with_columns(
                pl.col("atc").str.slice(*code_slice).alias(col)
                for col, code_slice in zip(temp_cols, code_slices)
            )
            .with_columns(
                pl.when(pl.col(col) != "")
                .then(
                    pl.lit(pfx)
                    + pl.col(col)
                    + (
                        pl.lit("//") + pl.col(col).replace_strict(code_to_desc, default=None)
                        if pfx == code_prefixes[0]
                        else pl.lit("")
                    )
                )
                .alias(col)
                for col, pfx in zip(temp_cols, code_prefixes)
            )
            .with_columns(unify_code_names(pl.col(temp_cols)))
        )

        if vocab is not None:
            df = apply_vocab_to_multitoken_codes(df, temp_cols, vocab)

        return (
            df.with_columns(code=pl.concat_list(temp_cols))
            .drop(*temp_cols, "atc")
            .explode("code")
            .drop_nulls("code")
        )


class LabData:
    @staticmethod
    @MatchAndRevise(prefix="LOINC/", apply_vocab=True)
    def retain_only_test_with_numeric_result(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("numeric_value").is_not_null())

    @staticmethod
    @MatchAndRevise(prefix="LOINC/", needs_counts=True, needs_vocab=True)
    def make_quantiles(
        df: pl.DataFrame, counts: dict[str, int] | None = None, vocab: list[str] | None = None
    ) -> pl.DataFrame:
        # TODO: we've run a simple analysis and decided to keep 200 most frequent labs
        # as the cover most of all the labs in the dataset
        return (
            df.with_columns(
                pl.concat_list("code", pl.lit("LOINC//Q//") + pl.col("code").str.slice(5)))
            .explode("code")
        )


class HCPCSData:
    @staticmethod
    @MatchAndRevise(prefix=["HCPCS/", "CPT4/"], apply_vocab=True)
    def unify_names(df: pl.DataFrame) -> pl.DataFrame:
        """This will just unify the code names."""
        return df


class ICUStayData:
    @staticmethod
    @MatchAndRevise(prefix="ICU_")
    def process(df: pl.DataFrame, *, num_quantiles: int = 10) -> pl.DataFrame:
        return df


class TransferData:
    @staticmethod
    @MatchAndRevise(prefix="TRANSFER_TO", apply_vocab=True)
    def retain_only_transfer_and_admit_types(df: pl.DataFrame) -> pl.DataFrame:
        return df


class BMIData:
    @staticmethod
    @MatchAndRevise(prefix="BMI")
    def make_quantiles(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(
                pl.col("text_value").cast(str).cast(float).alias("numeric_value"),
                pl.lit(None).alias("text_value"),
            )
            .filter(pl.col("numeric_value").is_between(10, 100))
            .with_columns(pl.concat_list(pl.lit("BMI"), pl.lit("BMI//Q")).alias("code"))
            .explode("code")
        )

    @staticmethod
    @MatchAndRevise(prefix=["BMI", "Q"])
    def join_token_and_quantile(df: pl.DataFrame) -> pl.DataFrame:
        q_following_bmi_mask = (pl.col("code") == "BMI").shift(1)
        return df.with_columns(
            code=pl.when(q_following_bmi_mask)
            .then(pl.lit("BMI//") + pl.col("code"))
            .when(pl.col("code") == "BMI")
            .then(None)
            .otherwise("code")
        ).drop_nulls("code")


class PatientFluidOutputData:
    @staticmethod
    @MatchAndRevise(prefix="SUBJECT_FLUID_OUTPUT//", needs_vocab=True)
    def make_quantiles(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        return df


class EdData:
    @staticmethod
    @MatchAndRevise(prefix="ED_REGISTRATION")
    def process_ed_registration(df: pl.DataFrame) -> pl.DataFrame:
        return df

    @staticmethod
    @MatchAndRevise(prefix="ACUITY")
    def process_ed_acuity(df: pl.DataFrame) -> pl.DataFrame:
        return df
