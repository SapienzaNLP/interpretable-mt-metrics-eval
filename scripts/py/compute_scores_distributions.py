import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mt_metrics_thresholds.definitions import (
    METRICS_DISTRIBUTIONS_DIR,
    METRIC_FILENAME_PATTERN,
    SUBMITTED_METRICS_DIRNAME,
    NEW_METRICS_DIRNAME,
)
from mt_metrics_thresholds.utils.plot import save_data_distribution_plot
from mt_metrics_thresholds.utils.wmt import (
    get_wmt_metric_name2scores,
    get_wmt_testset,
    get_grouped_metrics_scores,
)


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to compute and save plots of scores distributions."
    )
    parser.add_argument(
        "--testset-name",
        type=str,
        default="wmt23",
        help="Name of the WMT test set to use. Default: 'wmt23'.",
    )
    parser.add_argument(
        "--lp",
        type=str,
        default="zh-en",
        help="Language pair to consider. Default: 'zh-en'.",
    )
    parser.add_argument(
        "--ref-to-use",
        type=str,
        default="refA",
        help="Which human reference to consider. Default: 'refA'.",
    )
    parser.add_argument(
        "--include-human",
        action="store_true",
        help="Whether to include 'human' systems (i.e., reference translations) among systems.",
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Whether to include systems considered to be outliers.",
    )
    parser.add_argument(
        "--include-ref-to-use",
        action="store_true",
        help="Whether to include the employed reference system among systems to be scored (only for QE metrics).",
    )
    parser.add_argument(
        "--gold-name",
        type=str,
        default="mqm",
        help="Which human ratings to use as gold scores. Default: 'mqm'.",
    )
    parser.add_argument(
        "--scores-path",
        type=Path,
        help="Path to the .csv file containing scores. If passed, the above arguments are ignored, except 'lp'. If "
        "'lp' is not passed, all language pairs will be considered.",
    )
    parser.add_argument(
        "--scores-column",
        type=str,
        default="score",
        help="Name of the column in the 'scores-path' .csv file containing the scores. Default: 'score'.",
    )
    parser.add_argument(
        "--scores-name",
        type=str,
        default="da17-20",
        help="Name of the scores contained in the 'scores-path' .csv file. Default: 'da17-20'.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins to use in the histograms. Default: 30.",
    )
    parser.add_argument(
        "--new-metrics",
        action="store_true",
        help="Whether to use the new metrics info file.",
    )
    return parser


def compute_scores_distribution(
    scores_path: Path,
    scores_name: str,
    scores_column: str,
    bins: int,
    lp: Optional[str] = None,
) -> None:
    """
    Compute and save plots of the distribution of the scores in the input file.

    :param scores_path: Path to the .csv file containing scores (under a 'score' column).
    :param scores_name: Name of the scores contained in the `scores_path` .csv file.
    :param scores_column: Name of the column in the `scores_path` .csv file containing the scores.
    :param bins: Number of bins to use in the histograms.
    :param lp: If passed, only the scores for the given language pair will be considered. Default: None.
    """
    scores = []
    with open(scores_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if (
                lp is None or row.get("lp") == lp
            ):  # Filter by language pair if specified
                score = row.get(scores_column)
                if isinstance(score, str):
                    try:
                        scores.append(float(score))
                    except ValueError:
                        continue

    plot_title = (
        f"Distribution of Scores for {scores_name}"
        if lp is None
        else f"Distribution of Scores for {scores_name} ({lp})"
    )
    scores_column_dir = scores_path.parent / scores_column
    if not scores_column_dir.exists():
        scores_column_dir.mkdir(parents=True, exist_ok=True)
    file_path = (
        scores_column_dir / f"{scores_name}_scores_distribution.png"
        if lp is None
        else scores_column_dir / f"{scores_name}_{lp}_scores_distribution.png"
    )
    save_data_distribution_plot(
        scores,
        bins,
        plot_title,
        "Score",
        "Density",
        file_path,
        True,
    )


def compute_metric_distributions(
    metric_name2flattened_seg_scores: Dict[str, Tuple[List[float], List[float], bool]],
    testset_name: str,
    lp: str,
    bins: int,
    new_metrics: bool,
) -> None:
    """
    Compute and save plots of the distributions of the scores of the different MT metrics on the input WMT test set.

    :param metric_name2flattened_seg_scores: Dictionary from metric name to its flattened segment-level scores.
    :param testset_name: Name of the WMT test set.
    :param lp: Language pair to consider.
    :param bins: Number of bins to use in the histograms.
    :param new_metrics: Whether the checkpoints of some metrics are the news ones (wrt wmt23).
    """
    lp_dir = (
        METRICS_DISTRIBUTIONS_DIR
        / (SUBMITTED_METRICS_DIRNAME if not new_metrics else NEW_METRICS_DIRNAME)
        / testset_name
        / lp
    )
    if not lp_dir.exists():
        lp_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, (
        metric_seg_scores,
        gold_seg_scores,
        is_qe,
    ) in metric_name2flattened_seg_scores.items():
        save_data_distribution_plot(
            metric_seg_scores,
            bins,
            f"Distribution of Scores for {metric_name}",
            "Score",
            "Density",
            lp_dir
            / METRIC_FILENAME_PATTERN.format(metric_name=metric_name, extension="png"),
            True,
        )


if __name__ == "__main__":
    parser = read_arguments()
    args = parser.parse_args()

    if args.scores_path is not None:
        compute_scores_distribution(
            args.scores_path, args.scores_name, args.scores_column, args.bins, args.lp
        )
    else:
        testset = get_wmt_testset(args.testset_name, args.lp, True)
        compute_metric_distributions(
            get_grouped_metrics_scores(
                get_wmt_metric_name2scores(testset, args.ref_to_use, args.new_metrics),
                args.ref_to_use,
                args.include_human,
                args.include_outliers,
                args.include_ref_to_use,
                args.gold_name,
                "none",
                testset.DomainsPerSeg(),
                False,
                testset.human_sys_names,
                testset.outlier_sys_names,
            ),
            testset.name,
            testset.lp,
            args.bins,
            args.new_metrics,
        )
