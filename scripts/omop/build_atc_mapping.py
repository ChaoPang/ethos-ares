"""Builds the OMOP drug-to-ATC mapping used by `MedicationData.convert_to_atc`.

Drug codes in OMOP MEDS data (e.g., RxNorm/209387, NDC/16729029783) are mapped to 7-character
ATC codes using the OMOP vocabulary tables downloaded from Athena (https://athena.ohdsi.org):
non-standard codes (e.g., NDC) are first mapped to standard concepts via 'Maps to', and then
linked to ATC 5th level codes in two ways:

1. Direct primary links in CONCEPT_RELATIONSHIP (e.g., 'RxNorm - ATC'), which are used whenever
   a drug has them. Upward and secondary links ('... pr up', '... sec up/lat') point to
   combination classes (e.g., acetaminophen -> opioid combinations) and are ignored.
2. Otherwise, the closest ATC ancestors in CONCEPT_ANCESTOR. Direct links alone are not enough,
   as they exist almost only for ingredients and clinical drug forms, while drug exposures are
   mostly recorded as clinical or branded drugs. Only the closest ancestors are kept, because
   ingredients are descendants of every ATC combination class that contains them (e.g.,
   phenylephrine has ~80 ATC ancestors).

Drugs that still have more than `--max-atc-per-drug` ATC codes (e.g., generic multivitamins) are
left unmapped.

Usage:
    python scripts/omop/build_atc_mapping.py /path/to/omop_vocab drug_to_atc.csv \
        [--code-counts /path/to/code_counts.csv] [--max-atc-per-drug 10]
"""

import argparse
from pathlib import Path

import polars as pl

DRUG_VOCABULARIES = ["RxNorm", "RxNorm Extension", "NDC"]
PRIMARY_ATC_RELATIONSHIPS = [
    "RxNorm - ATC",
    "RxNorm - ATC pr lat",
    "Drug has drug class",
    "Mapped from",
]


def scan_athena_table(vocab_dir: Path, name: str) -> pl.LazyFrame:
    """Reads a vocabulary table stored either as Athena's tab-separated CSV (CONCEPT.csv) or as
    parquet, as a file or a directory of files (concept.parquet, concept/*.parquet)."""
    for table_name in [name, name.lower()]:
        if (fp := vocab_dir / f"{table_name}.csv").is_file():
            return pl.scan_csv(fp, separator="\t", quote_char=None, infer_schema=False)
        if (fp := vocab_dir / f"{table_name}.parquet").is_file():
            return pl.scan_parquet(fp)
        if (fp := vocab_dir / table_name).is_dir():
            return pl.scan_parquet(fp / "**/*.parquet")
    raise FileNotFoundError(f"Table '{name}' not found in {vocab_dir} (as CSV or parquet)")


def build_mapping(vocab_dir: Path, max_atc_per_drug: int = 10) -> pl.DataFrame:
    concepts = scan_athena_table(vocab_dir, "CONCEPT").select(
        "concept_id", "vocabulary_id", "concept_class_id", "concept_code"
    )
    drugs = concepts.filter(pl.col("vocabulary_id").is_in(DRUG_VOCABULARIES)).select(
        "concept_id", code=pl.col("vocabulary_id") + "/" + pl.col("concept_code")
    )
    atc_5th = concepts.filter(
        (pl.col("vocabulary_id") == "ATC") & (pl.col("concept_class_id") == "ATC 5th")
    ).select(ancestor_concept_id="concept_id", atc_code="concept_code")

    relationships = scan_athena_table(vocab_dir, "CONCEPT_RELATIONSHIP")
    maps_to = relationships.filter(pl.col("relationship_id") == "Maps to").select(
        concept_id="concept_id_1", standard_concept_id="concept_id_2"
    )
    # a drug is linked to ATC either directly or through the standard concept it maps to
    drug_to_standard = pl.concat(
        [
            drugs.select("code", standard_concept_id="concept_id"),
            drugs.join(maps_to, on="concept_id").select("code", "standard_concept_id"),
        ]
    ).unique()

    ancestors = scan_athena_table(vocab_dir, "CONCEPT_ANCESTOR").select(
        "ancestor_concept_id",
        standard_concept_id="descendant_concept_id",
        levels="min_levels_of_separation",
    )
    direct_links = (
        drug_to_standard.join(
            relationships.filter(pl.col("relationship_id").is_in(PRIMARY_ATC_RELATIONSHIPS)),
            left_on="standard_concept_id",
            right_on="concept_id_1",
        )
        .join(atc_5th, left_on="concept_id_2", right_on="ancestor_concept_id")
        .select("code", "atc_code")
    )
    closest_ancestors = (
        drug_to_standard.join(ancestors, on="standard_concept_id")
        .join(atc_5th, on="ancestor_concept_id")
        .filter(pl.col("levels") == pl.col("levels").min().over("code"))
        .join(direct_links.select("code"), on="code", how="anti")
        .select("code", "atc_code")
    )
    return (
        pl.concat([direct_links, closest_ancestors])
        .unique()
        .filter(pl.len().over("code") <= max_atc_per_drug)
        .sort("code", "atc_code")
        .collect()
    )


def report_coverage(mapping: pl.DataFrame, code_counts_fp: Path) -> None:
    counts = pl.read_csv(code_counts_fp).filter(
        pl.col("code").str.contains(f"^({'|'.join(DRUG_VOCABULARIES)})/")
    )
    count_col = counts.columns[1]
    mapped = counts.filter(pl.col("code").is_in(mapping["code"].unique()))
    print(
        f"Drug codes mapped to ATC: {len(mapped):,}/{len(counts):,} "
        f"({mapped[count_col].sum() / counts[count_col].sum():.1%} of drug events)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "vocab_dir", type=Path, help="Directory with the OMOP vocabulary tables (CSV or parquet)"
    )
    parser.add_argument("output_fp", type=Path, help="Output CSV file")
    parser.add_argument(
        "--code-counts", type=Path, help="code_counts.csv from a previous tokenization run, "
        "used to report how many drug codes and events are covered by the mapping"
    )
    parser.add_argument(
        "--max-atc-per-drug", type=int, default=10,
        help="Drugs with more closest ATC codes than this are left unmapped"
    )
    args = parser.parse_args()

    mapping = build_mapping(args.vocab_dir, args.max_atc_per_drug)
    args.output_fp.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_csv(args.output_fp)
    print(f"Wrote {len(mapping):,} mappings for {mapping['code'].n_unique():,} drug codes "
          f"to {args.output_fp}")

    if args.code_counts is not None:
        report_coverage(mapping, args.code_counts)
