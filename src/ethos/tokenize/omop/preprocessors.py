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


class DiagnosesData:
    @staticmethod
    @MatchAndRevise(prefix="ICD10CM")
    def prepare_codes_for_processing(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col.code.str.split_exact("/", 1)).with_columns(
            code=pl.lit("ICD//CM//10"), text_value=pl.col.code.struct[1]
        )

    @staticmethod
    @MatchAndRevise(prefix="ICD//CM//10", needs_vocab=True)
    def process_icd10(icd10_df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        from ..mappings import get_icd_cm_code_to_name_mapping

        code_to_name = get_icd_cm_code_to_name_mapping()
        temp_cols = ["part1", "part2", "part3"]
        code_prefixes = ["", "3-6//", "SFX//"]
        code_slices = [(0, 3), (3, 3), (6,)]

        df = (
            icd10_df.with_columns(
                pl.col("text_value").str.slice(*code_slice).alias(col)
                for col, code_slice in zip(temp_cols, code_slices)
            )
            .with_columns(pl.col(temp_cols[0]).replace_strict(code_to_name, default=None))
            .with_columns(
                pl.when(pl.col(col) != "")
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
            .drop(temp_cols)
            .explode("code")
            .drop_nulls("code")
        )


class ProcedureData:
    @staticmethod
    @MatchAndRevise(prefix=["ICD10PCS", "HCPCS"])
    def prepare_codes_for_processing(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(pl.col.code.str.split("/"))
            .with_columns(
                code=pl.lit("ICD//PCS//"), text_value=pl.col.code.list[1]
            )
        )

    @staticmethod
    @MatchAndRevise(prefix="ICD//PCS//10", needs_vocab=True)
    def process_icd10(icd10_df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        df = icd10_df.with_columns(
            pl.col("text_value").str.split_exact("", 6).alias("code")
        ).with_columns(
            code=pl.concat_list(
                pl.when(pl.col("code").struct[i] != "").then(
                    pl.lit("ICD//PCS//") + pl.col("code").struct[i]
                )
                for i in range(7)
            ).list.drop_nulls()
        )
        if vocab is not None:
            # all characters have to be in the vocab to keep the code
            df = df.filter(pl.col("code").list.eval(pl.element().is_in(vocab)).list.all())
        return df.explode("code").drop_nulls("code")


class MedicationData:
    @staticmethod
    @MatchAndRevise(prefix=["RxNorm", "NDC"], needs_vocab=True)
    def convert_to_atc(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        return df


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
