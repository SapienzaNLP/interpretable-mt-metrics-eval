import pickle
import random
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from mt_metrics_thresholds.definitions import (
    STABILITY_ANALYSIS_DIR,
    NEW_METRICS_DIRNAME,
    GOLD_SCORE_THRESHOLD_DIRNAME_PATTERN,
)
from mt_metrics_thresholds.optim import process_metric
from mt_metrics_thresholds.utils.wmt import (
    get_wmt_testset,
    get_wmt_metric_name2scores,
    get_grouped_metrics_scores,
    get_n_segs_with_gold_annotation,
)


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to compute and save data structures containing the results of the stability study."
    )
    parser.add_argument(
        "--precision-relative-weight",
        type=float,
        default=2,
        help="Importance weight of Precision over Recall. Default: 2.",
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
        "--step-size",
        type=int,
        default=50,
        help="How many segments to add at each step for the stability study. Default: 50.",
    )
    parser.add_argument(
        "--n-resamplings",
        type=int,
        default=5,
        help="Number of re-samplings to perform for each step. The final results for each step will be the mean. "
        "Default: 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to use for the segment sampling. Default: 42.",
    )
    return parser


def compute_and_save_stability_data(
    testset_name: str,
    lp: str,
    ref_to_use: str,
    include_human: bool,
    include_outliers: bool,
    include_ref_to_use: bool,
    gold_name: str,
    gold_score_threshold: float,
    precision_relative_weight: float,
    n_processes: int,
    step_size: int,
    n_resamplings: int,
    seed: int,
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
    :param precision_relative_weight: Importance weight of Precision over Recall.
    :param n_processes: Number of processes. If None, all available processors are used. Default: None.
    :param step_size: How many segments to add at each step for the stability study.
    :param n_resamplings: Number of re-samplings to perform for each step. The final step results will be the mean.
    :param seed: Seed to use for the segment sampling.
    """
    metrics_subset = {
        "metricx-23-xl",
        "metricx-23-qe-xl",
        "comet",
        "cometkiwi",
        "cometkiwi-xl",
        "sentinel-cand-mqm",
        "sentinel-src-mqm",
        "gemba-mqm",
        "matese",
        "matese-qe",
        "bertscore",
        "random-sysname",
        "bleu",
    }

    testset = get_wmt_testset(testset_name, lp, True)

    n_segs_with_gold_annotation = get_n_segs_with_gold_annotation(testset, gold_name)
    print("\n")
    print(f"# segments with gold annotation: {n_segs_with_gold_annotation}.")
    print("\n")
    sample_sizes = list(range(step_size, n_segs_with_gold_annotation + 1, step_size))
    if n_segs_with_gold_annotation % step_size != 0:
        sample_sizes.append(n_segs_with_gold_annotation)

    metric_name2scores = get_wmt_metric_name2scores(
        testset, ref_to_use, True, metrics_subset, gold_name
    )
    beta = 1 / (
        precision_relative_weight**0.5
    )  # Compute beta based on the precision relative weight
    sample_size2metrics_results = dict()

    random.seed(seed)
    np.random.seed(seed)
    for sample_size in tqdm(sample_sizes, desc="Iterating over sample sizes"):
        resampling_runs_results = []
        for _ in tqdm(range(n_resamplings), desc="Iterating over re-samplings"):
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
                sample_size=sample_size,
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

                resampling_runs_results.append(optimization_results)

        metrics_results = dict()
        for optimization_results in resampling_runs_results:
            for metric_name, optim_stats in optimization_results.items():
                if metric_name not in metrics_results:
                    metrics_results[metric_name] = dict()
                    for stat, value in optim_stats.items():
                        metrics_results[metric_name][stat] = [value]
                else:
                    for stat, value in optim_stats.items():
                        metrics_results[metric_name][stat].append(value)
        sample_size2metrics_results[sample_size] = metrics_results

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
    with open(stability_dir / f"sample_size2metrics_results.pickle", "wb") as f:
        pickle.dump(sample_size2metrics_results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = read_arguments()
    args = parser.parse_args()

    compute_and_save_stability_data(
        args.testset_name,
        args.lp,
        args.ref_to_use,
        args.include_human,
        args.include_outliers,
        args.include_ref_to_use,
        args.gold_name,
        args.gold_score_threshold,
        args.precision_relative_weight,
        args.n_processes,
        args.step_size,
        args.n_resamplings,
        args.seed,
    )
