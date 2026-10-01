from enum import StrEnum


class Task(StrEnum):
    HOSPITAL_MORTALITY = "hospital_mortality"
    HOSPITAL_MORTALITY_SINGLE = "hospital_mortality_single"
    READMISSION = "readmission"

    DRG_PREDICTION = "drg"
    SOFA_PREDICTION = "sofa"
    ICU_MORTALITY = "icu_mortality"

    ICU_READMISSION = "icu_readmission"

    ICU_ADMISSION = "icu_admission"
    ICU_ADMISSION_SINGLE = "icu_admission_single"

    # from the ED benchmark paper
    ED_HOSPITALIZATION = "ed_hospitalization"
    ED_CRITICAL_OUTCOME = "ed_critical_outcome"
    # the one below is called "ED reattendance" in the ED-Benchmark paper
    ED_REPRESENTATION = "ed_representation"

    # generic task driven by externally supplied ACES/MEDS labels, rather than
    # ground truth derived by scanning the record for what actually happened
    MEDS_LABEL = "meds_label"


class Reason(StrEnum):
    GOT_TOKEN = "token_of_interest"
    KEY_ERROR = "key_error"
    TIME_LIMIT = "time_limit"
