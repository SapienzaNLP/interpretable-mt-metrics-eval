import json
import logging
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Literal, Union, Optional

import numpy as np
from tqdm import tqdm

from mt_metrics_thresholds.definitions import (
    METRICS_RANKINGS_DIR,
    RANKINGS_FILENAME_PATTERN,
    GOLD_SCORE_PRECISION_THRESHOLDS_DIRNAME_PATTERN,
    RANK_CRITERION_PATTERN,
    SUBMITTED_METRICS_DIRNAME,
    NEW_METRICS_DIRNAME,
)
from mt_metrics_thresholds.optim import process_metric
from mt_metrics_thresholds.utils.wmt import (
    get_wmt_metric_name2scores,
    get_wmt_testset,
    get_grouped_metrics_scores,
    wmt_best_refs,
    lp2testset_names,
    official_wmt_settings,
    testset_name2lps,
    get_bio_metric_name2scores,
)


logger = logging.getLogger(__name__)


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to rank MT metrics according to the input meta-evaluation."
    )
    parser.add_argument(
        "--testset-names",
        type=str,
        nargs="+",
        help="Names (for WMT test sets) or paths of the test sets to be used.",
    )
    parser.add_argument(
        "--lps",
        type=str,
        nargs="+",
        help="Groups of language pairs (separated by commas) to use for the input test sets."
        "For example: --lps 'zh-en,en-ru' 'en-de'.",
    )
    parser.add_argument(
        "--rank-criterion-avg",
        type=str,
        choices=["macro", "micro"],
        default="macro",
        help="Averaging strategy to apply over distinct language pairs (if more than one is set). "
        "Allowed values: 'macro', 'micro'. Default: macro.",
    )
    parser.add_argument(
        "--refs-to-use",
        type=str,
        nargs="*",
        help="Groups of references (separated by commas) to use for the language pairs specified in the input WMT test "
        "sets (if any). For example: --refs-to-use 'refA,refB' 'refA'.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["data-filtering", "translation-reranking"],
        default="data-filtering",
        help="Meta-evaluation to run. Allowed values: 'data-filtering', 'translation-reranking'."
        "Default: 'data-filtering'.",
    )
    parser.add_argument(
        "--average-by",
        type=str,
        choices=["sys", "none"],
        help="Grouping strategy to use when running 'data-filtering' meta-evaluation. Allowed values: 'sys', 'none'.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="Beta real factor to use in the F_Beta computation when running 'data-filtering' meta-evaluation.",
    )
    parser.add_argument(
        "--include-human",
        type=str,
        nargs="*",
        help="Groups of boolean values (True/False), separated by commas, to indicate inclusion of 'human' systems "
        "(i.e., reference translations) among systems in the input WMT test sets (if any) for each language pair. "
        "For example: --include-human 'True,False' 'True'.",
    )
    parser.add_argument(
        "--include-outliers",
        type=str,
        nargs="*",
        help="Groups of boolean values (True/False), separated by commas, to indicate inclusion of systems considered "
        "to be outliers in the input WMT test sets (if any) for each language pair. "
        "For example: --include-outliers 'True,False' 'True'.",
    )
    parser.add_argument(
        "--include-ref-to-use",
        type=str,
        nargs="*",
        help="Groups of boolean values (True/False), separated by commas, to indicate whether to include the employed "
        "reference system among systems to be scored (only for QE metrics) in the input WMT test sets (if any) for"
        " each language pair. For example: --include-ref-to-use 'True,False' 'True'.",
    )
    parser.add_argument(
        "--gold-name",
        type=str,
        default="mqm",
        help="Which human ratings to use as gold scores from the input WMT test sets (if any). Default: 'mqm'.",
    )
    parser.add_argument(
        "--gold-score-threshold",
        type=float,
        default=0,
        help="Gold score threshold to consider in the optimization for 'data-filtering' meta-evaluation. Default: 0.",
    )
    parser.add_argument(
        "--precision-threshold",
        type=float,
        default=0,
        help="Precision threshold to consider in the optimization for 'data-filtering' meta-evaluation. Default: 0.",
    )
    parser.add_argument(
        "--thresholds-from-json",
        type=Path,
        help="Path to the json file containing thresholds for metrics (no optimization is run) for 'data-filtering' "
        "meta-evaluation. If passed, the '--precision-threshold' arg will be ignored.",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        help="Number of processes to use for parallelization. If left to None, the number of processes will be equal "
        "to the number of processors.",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        help="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    return parser


def rank_metrics(
    rank_criterion: Literal["f1", "ap", "auc_pr", "auc_pr_no_artificial_endpoint"],
    metric_name2grouped_seg_scores: Union[
        Dict[str, Tuple[List[float], List[float], bool]],
        Dict[str, Tuple[List[List[float]], List[List[float]], bool]],
    ],
    average_by: Literal["none", "item", "sys"],
    testset_name: str,
    lp: str,
    rank_criterion_avg: Literal["macro", "micro"],
    gold_name: str,
    gold_score_threshold: float,
    precision_threshold: float,
    beta: float,
    new_metrics: bool,
    valid_systems: Optional[List[str]] = None,
    valid_systems_qe: Optional[List[str]] = None,
    all_src_sents: Optional[List[str]] = None,
    all_ref_sents: Optional[List[str]] = None,
    sys2all_cand_sents: Optional[Dict[str, List[str]]] = None,
    metric_name2lp_offsets: Optional[Dict[str, List[int]]] = None,
    n_processes: Optional[int] = None,
    metric_name2thresholds: Optional[Dict[str, List[float]]] = None,
) -> None:
    """
    Rank the MT metrics according to the input parameters.

    :param rank_criterion: Ranking criterion. Allowed values: 'f1', 'ap', 'auc_pr', 'auc_pr_no_artificial_endpoint'.
    :param metric_name2grouped_seg_scores: Dictionary from metric name to its segment-level scores.
    :param average_by: What to average over when computing final scores.
    :param testset_name: Name of the WMT test set.
    :param lp: Language pair to consider.
    :param rank_criterion_avg: Avg strategy to apply over language pairs for ranking. Allowed values: 'macro', 'micro'.
    :param gold_name: Which human ratings to use as gold scores.
    :param gold_score_threshold: Gold score threshold to consider in the optimization.
    :param precision_threshold: Precision threshold to consider in the optimization.
    :param beta: Beta real factor to use in the F_Beta computation.
    :param new_metrics: Whether the metrics checkpoints are the news ones (wrt wmt23).
    :param valid_systems: Valid systems based on which items' scores are sorted (used for logging). Default: None.
    :param valid_systems_qe: Valid systems based on which QE items' scores are sorted (used for logging). Default: None.
    :param all_src_sents: List of source sentences (used for logging). Default: None.
    :param all_ref_sents: List of reference sentences (used for logging). Default: None.
    :param sys2all_cand_sents: Candidate translations dictionary (used for logging). Default: None.
    :param metric_name2lp_offsets: End indices of the data for each lp (when `lp` is set to 'all'). Default: None.
    :param n_processes: Number of processes. If None, all available processors are used. Default: None.
    :param metric_name2thresholds: Dictionary from metric name to the thresholds to use. Default: None.
    """

    if (
        rank_criterion != "f1"
        and rank_criterion != "ap"
        and rank_criterion != "auc_pr"
        and rank_criterion != "auc_pr_no_artificial_endpoint"
    ):
        raise ValueError(
            f"Invalid ranking criterion: {rank_criterion}! Allowed values: 'f1', 'ap', 'auc_pr', "
            f"'auc_pr_no_artificial_endpoint'."
        )

    if lp == "all" and metric_name2lp_offsets is None:
        raise ValueError(
            "When 'lp' is set to 'all', 'metric_name2lp_offsets' must be passed!"
        )

    if rank_criterion_avg != "macro" and rank_criterion_avg != "micro":
        raise ValueError(
            f"Invalid 'rank_criterion_avg parameter': {rank_criterion_avg}! Allowed values: 'macro', 'micro'."
        )

    if rank_criterion == "f1" and rank_criterion_avg == "micro":
        raise ValueError(
            "When 'rank_criterion' is set to 'f1', 'rank_criterion_avg' can only be set to 'macro'!"
        )

    percentile_ids = np.linspace(0, 100, 26).tolist()
    percentile_ids_for_fps_delta_distribution = [0, 20, 40, 60, 80, 100]

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = dict()

        for metric_name, (
            grouped_metric_seg_scores,
            grouped_gold_seg_scores,
            is_qe,
        ) in metric_name2grouped_seg_scores.items():
            if metric_name == gold_name:
                continue

            metric_valid_systems = valid_systems if not is_qe else valid_systems_qe

            if len(grouped_metric_seg_scores) != len(grouped_gold_seg_scores):
                if average_by == "none":
                    raise ValueError(
                        f"Metric {metric_name} has different number of scores than the gold {gold_name} scores! "
                        f"# metric scores = {len(grouped_metric_seg_scores)}, # gold {gold_name} scores = "
                        f"{len(grouped_gold_seg_scores)}."
                    )
                else:
                    raise ValueError(
                        f"Metric {metric_name} has different number of scored groups than the gold {gold_name} scores! "
                        f"# metric scored groups = {len(grouped_metric_seg_scores)}, # gold {gold_name} scored groups "
                        f"= {len(grouped_gold_seg_scores)}."
                    )

            future = executor.submit(
                process_metric,
                metric_name,
                grouped_metric_seg_scores,
                grouped_gold_seg_scores,
                average_by,
                testset_name,
                lp,
                metric_valid_systems,
                gold_score_threshold,
                precision_threshold,
                beta,
                new_metrics,
                True,
                percentile_ids,
                percentile_ids_for_fps_delta_distribution,
                all_src_sents,
                all_ref_sents,
                sys2all_cand_sents,
                metric_name2lp_offsets[metric_name] if lp == "all" else None,
                metric_name2thresholds[metric_name.lower()]
                if metric_name2thresholds is not None
                else None,
            )
            futures[future] = metric_name

        progress_bar = tqdm(total=len(futures), desc="Processing metrics")

        optimization_results, optimization_results_for_qe_ranking = dict(), dict()
        for future in as_completed(futures):
            metric_name = futures[future]
            _, optimization_result, optimization_result_for_qe_ranking = future.result()
            (
                optimization_results[metric_name],
                optimization_results_for_qe_ranking[metric_name],
            ) = (optimization_result, optimization_result_for_qe_ranking)
            progress_bar.update(1)

        progress_bar.close()

    ranked_metrics = sorted(
        optimization_results.items(),
        key=lambda x: x[1][
            RANK_CRITERION_PATTERN.format(
                rank_criterion_avg=rank_criterion_avg, rank_criterion=rank_criterion
            )
        ],
        reverse=True,
    )

    rankings_dir = (
        METRICS_RANKINGS_DIR
        / (SUBMITTED_METRICS_DIRNAME if not new_metrics else NEW_METRICS_DIRNAME)
        / testset_name
        / lp
        / (
            RANK_CRITERION_PATTERN.format(
                rank_criterion_avg=rank_criterion_avg, rank_criterion=rank_criterion
            )
            if lp == "all"
            else rank_criterion
        )
        / GOLD_SCORE_PRECISION_THRESHOLDS_DIRNAME_PATTERN.format(
            gold_score_threshold=gold_score_threshold,
            precision_threshold=precision_threshold,
        )
    )
    if not rankings_dir.exists():
        rankings_dir.mkdir(parents=True, exist_ok=True)
    with open(
        rankings_dir / RANKINGS_FILENAME_PATTERN.format(average_by=average_by), "w"
    ) as file:
        json.dump(dict(ranked_metrics), file, indent=4)

    if average_by == "item":
        ranked_metrics = sorted(
            optimization_results_for_qe_ranking.items(),
            key=lambda x: x[1]["macro_qe_ranking_precision"],
            reverse=True,
        )

        rankings_dir = (
            METRICS_RANKINGS_DIR
            / (SUBMITTED_METRICS_DIRNAME if not new_metrics else NEW_METRICS_DIRNAME)
            / testset_name
            / lp
            / "macro_qe_ranking_precision"
        )
        if not rankings_dir.exists():
            rankings_dir.mkdir(parents=True, exist_ok=True)
        with open(rankings_dir / "ranking.json", "w") as file:
            json.dump(dict(ranked_metrics), file, indent=4)


if __name__ == "__main__":
    parser = read_arguments()
    args = parser.parse_args()

    numeric_log_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError(f"Invalid log level passed in input: {numeric_log_level}")
    logging.basicConfig(
        level=numeric_log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if len(args.testset_names) != len(args.lps):
        raise ValueError(
            f"The number of test sets and language pairs must be the same! # test sets: {len(args.testset_names)}, "
            f"# lps: {len(args.lps)}."
        )

    for testset_name, testset_lps in zip(args.testset_names, args.lps):
        testset_path = Path(testset_name)

        for lp in testset_lps.split(","):
            if not testset_path.exists():
                testset = get_wmt_testset(testset_name, args.lp, True)

    metric_name2lp_offsets = None
    metric_name2thresholds = None
    if args.lp == "all":
        metric_name2grouped_seg_scores, metric_name2lp_offsets = dict(), dict()

        lp2testset_names_to_use = (
            lp2testset_names
            if args.testset == "all"
            else {lp: [args.testset_name] for lp in testset_name2lps[args.testset_name]}
        )

        for lp, testset_names in lp2testset_names_to_use.items():
            for testset_name in testset_names:
                testset = get_wmt_testset(testset_name, lp, True)
                lp_metric_name2grouped_seg_scores = get_grouped_metrics_scores(
                    get_wmt_metric_name2scores(
                        testset, wmt_best_refs[testset_name][lp], args.new_metrics
                    ),
                    wmt_best_refs[testset_name][lp],
                    official_wmt_settings[testset_name][lp]["include_human"]
                    if args.official_wmt_settings
                    else args.include_human,
                    official_wmt_settings[testset_name][lp]["include_outliers"]
                    if args.official_wmt_settings
                    else args.include_outliers,
                    False if args.official_wmt_settings else args.include_ref_to_use,
                    args.gold_name,
                    args.average_by,
                    testset.DomainsPerSeg(),
                    False,
                    testset.human_sys_names,
                    testset.outlier_sys_names,
                )

                for metric_name, (
                    grouped_metric_seg_scores,
                    grouped_gold_seg_score,
                    is_qe,
                ) in lp_metric_name2grouped_seg_scores.items():
                    if metric_name not in metric_name2grouped_seg_scores:
                        metric_name2grouped_seg_scores[metric_name] = [[], [], is_qe]
                    metric_name2grouped_seg_scores[metric_name][
                        0
                    ] += grouped_metric_seg_scores
                    metric_name2grouped_seg_scores[metric_name][
                        1
                    ] += grouped_gold_seg_score

            for metric_name, (
                grouped_metric_seg_scores,
                _,
                _,
            ) in metric_name2grouped_seg_scores.items():
                if metric_name not in metric_name2lp_offsets:
                    metric_name2lp_offsets[metric_name] = []
                metric_name2lp_offsets[metric_name].append(
                    len(grouped_metric_seg_scores) - 1
                )

        for metric_name, data in metric_name2grouped_seg_scores.items():
            metric_name2grouped_seg_scores[metric_name] = tuple(data)

    else:
        if args.testset_name == "all":
            metric_name2grouped_seg_scores = dict()
            for testset_name in ["wmt22", "wmt23"]:
                testset = get_wmt_testset(testset_name, args.lp, True)

                testset_metric_name2grouped_seg_scores = get_grouped_metrics_scores(
                    get_wmt_metric_name2scores(
                        testset, wmt_best_refs[testset_name][args.lp], args.new_metrics
                    ),
                    wmt_best_refs[testset_name][args.lp],
                    official_wmt_settings[testset_name][args.lp]["include_human"]
                    if args.official_wmt_settings
                    else args.include_human,
                    official_wmt_settings[testset_name][args.lp]["include_outliers"]
                    if args.official_wmt_settings
                    else args.include_outliers,
                    False if args.official_wmt_settings else args.include_ref_to_use,
                    args.gold_name,
                    args.average_by,
                    testset.DomainsPerSeg(),
                    False,
                    testset.human_sys_names,
                    testset.outlier_sys_names,
                )

                for metric_name, (
                    grouped_metric_seg_scores,
                    grouped_gold_seg_score,
                    is_qe,
                ) in testset_metric_name2grouped_seg_scores.items():
                    if metric_name not in metric_name2grouped_seg_scores:
                        metric_name2grouped_seg_scores[metric_name] = [[], [], is_qe]
                    metric_name2grouped_seg_scores[metric_name][
                        0
                    ] += grouped_metric_seg_scores
                    metric_name2grouped_seg_scores[metric_name][
                        1
                    ] += grouped_gold_seg_score

            for metric_name, data in metric_name2grouped_seg_scores.items():
                metric_name2grouped_seg_scores[metric_name] = tuple(data)

        else:
            domains_per_seg, human_sys_names, outlier_sys_names = None, None, None
            if args.testset_name == "bio":
                metric_name2scores = get_bio_metric_name2scores(args.lp)
            else:
                metrics_subset = None

                if args.thresholds_from_json is not None:
                    with open(args.thresholds_from_json, "r") as file:
                        metric_name2thresholds = {
                            metric_name.lower(): [
                                optim_metric_dict["optimal_metric_threshold"]
                            ]
                            for metric_name, optim_metric_dict in json.load(
                                file
                            ).items()
                        }
                    metrics_subset = set(metric_name2thresholds)

                testset = get_wmt_testset(args.testset_name, args.lp, True)
                metric_name2scores = get_wmt_metric_name2scores(
                    testset,
                    args.ref_to_use,
                    args.new_metrics,
                    metrics_subset,
                    args.gold_name,
                )
                domains_per_seg, human_sys_names, outlier_sys_names = (
                    testset.DomainsPerSeg(),
                    testset.human_sys_names,
                    testset.outlier_sys_names,
                )

            metric_name2grouped_seg_scores = get_grouped_metrics_scores(
                metric_name2scores,
                args.ref_to_use,
                args.include_human,
                args.include_outliers,
                args.include_ref_to_use,
                args.gold_name,
                args.average_by,
                domains_per_seg,
                args.log_tps_fps_fns_tns,
                human_sys_names,
                outlier_sys_names,
                args.testset_name == "bio",
            )

            if args.log_tps_fps_fns_tns:
                (
                    metric_name2grouped_seg_scores,
                    valid_systems,
                    valid_systems_qe,
                ) = metric_name2grouped_seg_scores
                all_src_sents, all_ref_sents = (
                    testset.src,
                    testset.all_refs[args.ref_to_use],
                )
                sys2all_cand_sents = testset.sys_outputs

    rank_metrics(
        args.rank_criterion,
        metric_name2grouped_seg_scores,
        args.average_by,
        args.testset_name,
        args.lp,
        args.rank_criterion_avg,
        args.gold_name,
        args.gold_score_threshold,
        args.precision_threshold if metric_name2thresholds is None else 0,
        args.beta,
        args.new_metrics,
        valid_systems,
        valid_systems_qe,
        all_src_sents,
        all_ref_sents,
        sys2all_cand_sents,
        metric_name2lp_offsets,
        args.n_processes,
        metric_name2thresholds,
    )
