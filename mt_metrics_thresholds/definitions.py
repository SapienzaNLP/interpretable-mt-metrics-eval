from pathlib import Path

# Paths

# General
ROOT_DIR = Path(__file__).parent.parent

# Dirs
METRICS_DATA_DIR = ROOT_DIR / "metrics_data"
SUBMITTED_METRICS_DIRNAME = "submitted_metrics"
NEW_METRICS_DIRNAME = "new_metrics"
METRICS_DISTRIBUTIONS_DIR = METRICS_DATA_DIR / "distributions"
METRICS_INFO_DIR = METRICS_DATA_DIR / "info"
METRICS_OUTPUTS_DIR = "metrics_outputs"
HUMAN_SYS_NAMES_FILE = "human_sys_names.txt"
OUTLIER_SYS_NAMES_FILE = "outlier_sys_names.txt"
HUMAN_SCORE_NAMES_FILE = "human_score_names.txt"
METRICS_PR_PLOTS_DIR = METRICS_DATA_DIR / "pr_plots"
METRICS_RANKINGS_DIR = METRICS_DATA_DIR / "rankings"
METRICS_FPS_PLOTS_DIR = METRICS_DATA_DIR / "fps_plots"
METRICS_FPS_DELTAS_DIR = METRICS_DATA_DIR / "fps_deltas"
METRICS_DUMPS = METRICS_DATA_DIR / "dumps"
DA_SQM_CORRS_DIR = METRICS_DATA_DIR / "da_sqm_correlations"
REGRESSION_ANALYSIS_DIR = METRICS_DATA_DIR / "regression_analysis"
STABILITY_ANALYSIS_DIR = METRICS_DATA_DIR / "stability_analysis"
MBR_PREDICTIONS_DIR = METRICS_DATA_DIR / "mbr"

# Annotated data
DA_DIR = ROOT_DIR / "annotated_data" / "train" / "da"

# Filenames

METRICS_INFO_FILENAME = "metrics_info.tsv"
NEW_METRICS_INFO_FILENAME = "new_metrics_info.tsv"

SEG_SCORES_FILENAME = "seg_scores.pickle"
SYS_SCORES_FILENAME = "sys_scores.pickle"

GOLD_SCORE_THRESHOLD_DIRNAME_PATTERN = "gold_score_threshold_{gold_score_threshold}"
AVERAGE_BY_DIRNAME_PATTERN = "{average_by}_grouping"
GOLD_SCORE_PRECISION_THRESHOLDS_DIRNAME_PATTERN = (
    "gold_score_threshold_{gold_score_threshold}__precision_threshold_{"
    "precision_threshold}"
)

RANKINGS_FILENAME_PATTERN = "{average_by}_grouping_metrics_rankings.json"

METRIC_FILENAME_PATTERN = "{metric_name}.{extension}"

# Metrics
DISCRETE_METRICS = {"MaTESe", "MaTESe-QE", "GEMBA-MQM"}

# Special Strings
RANK_CRITERION_PATTERN = "{rank_criterion_avg}_{rank_criterion}"

STABILITY_STUDY_N_SEGS_STEP = 50
