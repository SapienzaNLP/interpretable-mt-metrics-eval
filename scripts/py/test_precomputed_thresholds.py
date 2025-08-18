import pickle
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from mt_metrics_thresholds.definitions import (
    STABILITY_ANALYSIS_DIR,
    GOLD_SCORE_THRESHOLD_DIRNAME_PATTERN,
    NEW_METRICS_DIRNAME,
)
from mt_metrics_thresholds.optim import process_metric
from mt_metrics_thresholds.utils.wmt import (
    get_wmt_testset,
    get_wmt_metric_name2scores,
    get_grouped_metrics_scores,
)


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to perform the stability study from pre-computed thresholds."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Beta real factor to use in the F_Beta computation. Default: 0.5.",
    )
    parser.add_argument(
        "--testset-name",
        type=str,
        choices=["wmt22", "wmt23"],
        default="wmt23",
        help="Name of the WMT test set to use. Allowed values: 'wmt22' and 'wmt23'. Default: 'wmt23'.",
    )
    parser.add_argument(
        "--lp",
        type=str,
        default="zh-en",
        help="Language pair to consider in the test set passed in input.",
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
        "--gold-score-threshold",
        type=float,
        default=-1,
        help="Gold score threshold to consider in the optimization. Default: -1.",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        help="Number of processes to use for parallelization. If left to None, the number of processes will be equal "
        "to the number of processors.",
    )
    parser.add_argument(
        "--thresholds-from-pickle",
        type=Path,
        help="Path to the pickle file containing thresholds for metrics (no optim is run).",
    )
    parser.add_argument(
        "--dev-set-name",
        type=str,
        default="wmt22",
        help="Name of the dev set where the thresholds have been optimizes. Default: 'wmt22'.",
    )
    return parser


def test_precomputed_thresholds(
    testset_name: str,
    lp: str,
    ref_to_use: str,
    include_human: bool,
    include_outliers: bool,
    include_ref_to_use: bool,
    gold_name: str,
    gold_score_threshold: float,
    beta: float,
    n_processes: int,
    thresholds_from_pickle: Path,
    dev_set_name: str,
) -> None:
    """
    Compute and save data structures containing the results of the stability study.

    :param testset_name: Name of the WMT test set to use.
    :param lp: Language pair to consider in the test set passed in input.
    :param ref_to_use: Which reference to use for reference-based metrics.
    :param include_human: Whether to include human systems (i.e., reference translations) among systems.
    :param include_outliers: Whether to include systems considered to be outliers.
    :param include_ref_to_use: Whether to include the reference system (for QE metrics).
    :param gold_name: Which human ratings to use as gold scores.
    :param gold_score_threshold: Gold score threshold to consider in the optimization.
    :param beta: Beta real factor to use in the F_Beta computation.
    :param n_processes: Number of processes. If None, all available processors are used. Default: None.
    :param thresholds_from_pickle: Path to the pickle file containing thresholds for metrics (no optim is run).
    :param dev_set_name: Name of the dev set where the thresholds have been optimized.
    """
    with open(thresholds_from_pickle, "rb") as handle:
        sample_size2metrics_results = pickle.load(handle)
    metrics_subset = set(
        [
            metric_name.lower()
            for metric_name in next(iter(sample_size2metrics_results.values()))
        ]
    )

    testset = get_wmt_testset(testset_name, lp, True)

    metric_name2scores = get_wmt_metric_name2scores(
        testset, ref_to_use, True, metrics_subset, gold_name
    )

    for sample_size, metric_name2results in tqdm(
        sample_size2metrics_results.items(), desc="Iterating over sample sizes"
    ):
        metric_name2grouped_seg_scores = get_grouped_metrics_scores(
            metric_name2scores,
            ref_to_use,
            include_human,
            include_outliers,
            include_ref_to_use,
            gold_name,
            "sys",
            human_sys_names=testset.human_sys_names,
            outlier_sys_names=testset.outlier_sys_names,
        )

        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = dict()

            for metric_name, (
                grouped_metric_seg_scores,
                grouped_gold_seg_scores,
                is_qe,
            ) in metric_name2grouped_seg_scores.items():
                if metric_name == gold_name:
                    continue

                if len(grouped_metric_seg_scores) != len(grouped_gold_seg_scores):
                    raise ValueError(
                        f"Metric {metric_name} has different number of scored systems than the gold {gold_name} "
                        f"scores! # metric scored systems = {len(grouped_metric_seg_scores)}, "
                        f"# gold {gold_name} scored systems = {len(grouped_gold_seg_scores)}."
                    )

                future = executor.submit(
                    process_metric,
                    metric_name,
                    grouped_metric_seg_scores,
                    grouped_gold_seg_scores,
                    "sys",
                    testset_name,
                    lp,
                    [],
                    gold_score_threshold,
                    0,
                    beta,
                    True,
                    False,
                    thresholds_to_use=metric_name2results[metric_name][
                        "optimal_metric_threshold"
                    ],
                    return_results_with_all_thresholds_to_use=True,
                )
                futures[future] = metric_name

            optimization_results = dict()
            for future in as_completed(futures):
                metric_name = futures[future]
                (
                    _,
                    optimization_result,
                    _,
                ) = future.result()
                optimization_results[metric_name] = optimization_result

        sample_size2metrics_results[sample_size] = optimization_results

    stability_dir = (
        STABILITY_ANALYSIS_DIR
        / NEW_METRICS_DIRNAME
        / testset.name
        / testset.lp
        / GOLD_SCORE_THRESHOLD_DIRNAME_PATTERN.format(
            gold_score_threshold=gold_score_threshold
        )
    )

    if not stability_dir.exists():
        stability_dir.mkdir(parents=True, exist_ok=True)
    with open(
        stability_dir
        / f"sample_size2metrics_results_thresholds_from_{dev_set_name}.pickle",
        "wb",
    ) as f:
        pickle.dump(sample_size2metrics_results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = read_arguments()
    args = parser.parse_args()

    test_precomputed_thresholds(
        args.testset_name,
        args.lp,
        args.ref_to_use,
        args.include_human,
        args.include_outliers,
        args.include_ref_to_use,
        args.gold_name,
        args.gold_score_threshold,
        args.beta,
        args.n_processes,
        args.thresholds_from_pickle,
        args.dev_set_name,
    )
