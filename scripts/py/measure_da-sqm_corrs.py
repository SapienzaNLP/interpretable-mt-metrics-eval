from argparse import ArgumentParser
from pprint import pp
from typing import List, Dict, Tuple, Set, Callable, Any, Literal, Optional

import scipy.stats
from mt_metrics_eval import data, stats

from mt_metrics_thresholds.definitions import DA_SQM_CORRS_DIR
from mt_metrics_thresholds.utils.wmt import (
    get_wmt_testset,
    get_wmt_metric_name2scores,
)

corr_fcn2name = {
    scipy.stats.kendalltau: "KendallTau",
    stats.KendallWithTiesOpt: "KendallWithTiesOpt",
    scipy.stats.pearsonr: "Pearson",
}


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to compute correlations between DA-SQM and MQM annotations on WMT-23."
    )
    parser.add_argument(
        "--lp",
        type=str,
        default="zh-en",
        help="Language pair to consider in WMT-23 (DA-SQM overlaps with MQM only in zh-en and en-de). "
        "Default: 'zh-en'.",
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
        "--gold-name",
        type=str,
        default="mqm",
        help="Which human ratings to use as gold scores. Default: 'mqm'.",
    )
    parser.add_argument(
        "--new-metrics",
        action="store_true",
        help=" Whether the checkpoints of some metrics are the news ones (wrt wmt23).",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Sample rate to pass to tau_optimization for 'KendallWithTiesOpt'. Default: 1.0.",
    )
    return parser


def get_correlation_value(
    testset: data.EvalSet,
    gold_scores: Dict[str, List[float]],
    metric_scores: Dict[str, List[float]],
    sys_names: Set[str],
    corr_fcn: Callable,
    average_by: Literal["none", "item", "sys"],
    **corr_fcn_args: Any,
) -> float:
    """
    Compute the correlation between two groups of scores.

    :param testset: The WMT test set to use.
    :param gold_scores: Gold scores.
    :param metric_scores: Metric scores.
    :param sys_names: MT system names to consider in the correlation.
    :param corr_fcn: Correlation function to employ.
    :param average_by: What to average over when computing the correlation value. Allowed values: 'none', 'item', 'sys'.
    :param corr_fcn_args: Optional extra arguments for corr_fcn.

    :return: The computed correlation value.
    """
    if corr_fcn not in [
        scipy.stats.kendalltau,
        stats.KendallWithTiesOpt,
        scipy.stats.pearsonr,
    ]:
        raise ValueError(
            "Correlation function not allowed for 'get_correlation_value' method. Choose from scipy.stats.kendalltau, "
            "stats.KendallWithTiesOpt, or scipy.stats.pearsonr."
        )

    correlation_obj = testset.Correlation(
        gold_scores,
        metric_scores,
        sys_names,
    )
    corr_wrapper = stats.AverageCorrelation(
        corr_fcn,
        correlation_obj.num_sys,
        average_by=average_by,
        filter_nones=correlation_obj.none_count,
        replace_nans_with_zeros=False,
        **corr_fcn_args,
    )
    corr_value = corr_wrapper(
        correlation_obj.gold_scores, correlation_obj.metric_scores
    )[0]
    return corr_value


def save_da_sqm_correlations_report(
    metric_name2scores: Dict[
        str, Tuple[Dict[str, List[float]], Dict[str, List[float]], bool, Optional[str]]
    ],
    testset: data.EvalSet,
    gold_name: str,
    ref_to_use: str,
    include_human: bool,
    include_outliers: bool,
    sample_rate: float,
) -> None:
    """
    Compute correlations of DA-SQM and MT metrics with `gold_name` annotations on WMT-23 and save them in a report file.

    :param metric_name2scores: Dictionary containing metric scores for segment-level and system-level annotations.
    :param testset: WMT test set to use.
    :param gold_name: Name of the gold metric to use.
    :param ref_to_use: Which human reference to consider.
    :param include_human: Whether to include 'human' systems (i.e., reference translations) among systems.
    :param include_outliers: Whether to include systems considered to be outliers.
    :param sample_rate: Sample rate to pass to tau_optimization for 'KendallWithTiesOpt'.
    """
    # Get DA-SQM and gold_name annotations
    da_sqm_sys2seg_scores, da_sqm_sys2score, _, _ = metric_name2scores["da-sqm"]
    gold_sys2seg_scores, gold_sys2score, _, _ = metric_name2scores[gold_name]

    sys_names = set(da_sqm_sys2seg_scores).intersection(set(gold_sys2seg_scores))
    sys_names.discard(ref_to_use)  # For ref-based metrics
    if not include_human:
        for sys in testset.human_sys_names:
            sys_names.discard(sys)
    if not include_outliers:
        for sys in testset.outlier_sys_names:
            sys_names.discard(sys)

    print("\n")
    print(f"# MT Systems = {len(sys_names)}.")
    print("\n")

    metric_name2filtered_scores = dict()
    for i, (metric_name, (sys2seg_scores, sys2score, _, _)) in enumerate(
        metric_name2scores.items()
    ):
        metric_name2filtered_scores[metric_name] = (dict(), dict())
        for sys in sys_names:
            if not (
                len(da_sqm_sys2seg_scores[sys])
                == len(gold_sys2seg_scores[sys])
                == len(sys2seg_scores[sys])
            ):
                raise ValueError(
                    f"DA-SQM, MQM, and {metric_name} annotations for system '{sys}' have different lengths! "
                    f"DA-SQM length :{len(da_sqm_sys2seg_scores[sys])}, MQM length: {len(gold_sys2seg_scores[sys])}, "
                    f"{metric_name} length: {len(sys2seg_scores[sys])}."
                )

            metric_name2filtered_scores[metric_name][0][sys] = []
            for da_sqm_seg_score, gold_seg_score, metric_seg_score in zip(
                da_sqm_sys2seg_scores[sys],
                gold_sys2seg_scores[sys],
                sys2seg_scores[sys],
            ):
                if da_sqm_seg_score is not None and gold_seg_score is not None:
                    assert metric_seg_score is not None
                    metric_name2filtered_scores[metric_name][0][sys].append(
                        metric_seg_score
                    )

            if i == 0:
                print(
                    f"# Segments for system '{sys}' = {len(metric_name2filtered_scores[metric_name][0][sys])}."
                )

            metric_name2filtered_scores[metric_name][1][sys] = [
                sum(metric_name2filtered_scores[metric_name][0][sys])
                / len(metric_name2filtered_scores[metric_name][0][sys])
            ]

    seg_level_corr_fcn_name2average_by_results, metric_name2pearson_sys_level_corr = {
        "KendallTau": {"none": dict(), "item": dict(), "sys": dict()},
        "KendallWithTiesOpt": {"item": dict()},
        "Pearson": {"none": dict(), "item": dict(), "sys": dict()},
    }, dict()
    for metric_name, (
        filtered_sys2seg_scores,
        filtered_sys2score,
    ) in metric_name2filtered_scores.items():
        if metric_name == gold_name:
            continue

        for corr_function in [
            scipy.stats.kendalltau,
            stats.KendallWithTiesOpt,
            scipy.stats.pearsonr,
        ]:
            if corr_function != stats.KendallWithTiesOpt:
                seg_level_corr_fcn_name2average_by_results[
                    corr_fcn2name[corr_function]
                ]["none"][metric_name] = get_correlation_value(
                    testset,
                    metric_name2filtered_scores[gold_name][0],
                    filtered_sys2seg_scores,
                    sys_names,
                    corr_function,
                    "none",
                )
            corr_fcn_args = dict()
            if corr_function == stats.KendallWithTiesOpt:
                corr_fcn_args["sample_rate"] = sample_rate
            seg_level_corr_fcn_name2average_by_results[corr_fcn2name[corr_function]][
                "item"
            ][metric_name] = get_correlation_value(
                testset,
                metric_name2filtered_scores[gold_name][0],
                filtered_sys2seg_scores,
                sys_names,
                corr_function,
                "item",
                **corr_fcn_args,
            )
            if corr_function != stats.KendallWithTiesOpt:
                seg_level_corr_fcn_name2average_by_results[
                    corr_fcn2name[corr_function]
                ]["sys"][metric_name] = get_correlation_value(
                    testset,
                    metric_name2filtered_scores[gold_name][0],
                    filtered_sys2seg_scores,
                    sys_names,
                    corr_function,
                    "sys",
                )

        metric_name2pearson_sys_level_corr[metric_name] = get_correlation_value(
            testset,
            metric_name2filtered_scores[gold_name][1],
            filtered_sys2score,
            sys_names,
            scipy.stats.pearsonr,
            "none",
        )

    for (
        corr_fcn_name,
        levels_dict,
    ) in seg_level_corr_fcn_name2average_by_results.items():
        for level, metrics_dict in levels_dict.items():
            sorted_metrics_dict = dict(
                sorted(metrics_dict.items(), key=lambda item: item[1], reverse=True)
            )
            seg_level_corr_fcn_name2average_by_results[corr_fcn_name][
                level
            ] = sorted_metrics_dict

    sorted_metric_name2pearson_sys_level_corr = dict(
        sorted(
            metric_name2pearson_sys_level_corr.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    lp_dir = DA_SQM_CORRS_DIR / testset.lp
    lp_dir.mkdir(parents=True, exist_ok=True)
    with open(lp_dir / "correlations_report.txt", "w") as f:
        f.write(
            f"Correlations of DA-SQM and MT metrics with MQM on WMT-23 for language pair '{testset.lp}'\n"
            f"Segment-level correlations:\n"
            f"  - KendallTau (none): {seg_level_corr_fcn_name2average_by_results['KendallTau']['none']}\n"
            f"  - KendallTau (item): {seg_level_corr_fcn_name2average_by_results['KendallTau']['item']}\n"
            f"  - KendallTau (sys): {seg_level_corr_fcn_name2average_by_results['KendallTau']['sys']}\n"
            f"  - KendallWithTiesOpt (item): {seg_level_corr_fcn_name2average_by_results['KendallWithTiesOpt']['item']}"
            f"\n"
            f"  - Pearson (none): {seg_level_corr_fcn_name2average_by_results['Pearson']['none']}\n"
            f"  - Pearson (item): {seg_level_corr_fcn_name2average_by_results['Pearson']['item']}\n"
            f"  - Pearson (sys): {seg_level_corr_fcn_name2average_by_results['Pearson']['sys']}\n"
            f"System-level Pearson correlations: {sorted_metric_name2pearson_sys_level_corr}\n"
        )


if __name__ == "__main__":
    parser = read_arguments()
    args = parser.parse_args()

    testset = get_wmt_testset("wmt23", args.lp, True)
    save_da_sqm_correlations_report(
        get_wmt_metric_name2scores(testset, args.ref_to_use, args.new_metrics),
        testset,
        args.gold_name,
        args.ref_to_use,
        args.include_human,
        args.include_outliers,
        args.sample_rate,
    )
