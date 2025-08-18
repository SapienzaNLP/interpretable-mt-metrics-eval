from argparse import ArgumentParser
from typing import Dict, List, Tuple, Union

from mt_metrics_eval import data

from mt_metrics_thresholds.definitions import (
    METRICS_DUMPS,
    METRIC_FILENAME_PATTERN,
    SUBMITTED_METRICS_DIRNAME,
    NEW_METRICS_DIRNAME,
)
from mt_metrics_thresholds.utils.wmt import (
    get_wmt_metric_name2scores,
    get_wmt_testset,
    get_grouped_metrics_scores,
)


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to compute and save the dumps of the scores given by the MT metrics on a WMT test set."
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
        help="Language pair to consider in the test set passed in input. Default: 'zh-en'.",
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
        "--new-metrics",
        action="store_true",
        help="Whether to use the new metrics info file.",
    )
    return parser


def save_metrics_dump(
    metric_name2item_grouped_seg_scores: Dict[
        str, Tuple[List[List[float]], List[List[float]], bool]
    ],
    testset: data.EvalSet,
    ref_to_use: str,
    gold_name: str,
    valid_systems: List[str],
    valid_systems_qe: List[str],
    new_metrics: bool = False,
) -> None:
    """
    Compute and save a dump of the scores given by the different MT metrics on the input WMT test set.

    :param metric_name2item_grouped_seg_scores: Dictionary from metric name to its item-grouped segment-level scores.
    :param testset: WMT test set to consider.
    :param ref_to_use: Which human reference to consider.
    :param gold_name: Which human ratings to use as gold scores.
    :param valid_systems: List of valid system names based on which the scores for each item are sorted.
    :param valid_systems_qe: List of valid system names based on which the QE scores for each item are sorted.
    :param new_metrics: Whether the checkpoints of some metrics are the news ones (wrt wmt23).
    """
    lp_dir = (
        METRICS_DUMPS
        / (SUBMITTED_METRICS_DIRNAME if not new_metrics else NEW_METRICS_DIRNAME)
        / testset.name
        / testset.lp
    )
    if not lp_dir.exists():
        lp_dir.mkdir(parents=True, exist_ok=True)

    all_src_sents, all_ref_sents = testset.src, testset.all_refs[ref_to_use]
    sys2all_cand_sents = testset.sys_outputs
    for (
        metric_name,
        (item_grouped_metric_scores, item_grouped_gold_seg_scores, is_qe),
    ) in metric_name2item_grouped_seg_scores.items():
        if metric_name == gold_name:
            continue

        if not (
            len(item_grouped_metric_scores) == len(all_src_sents) == len(all_ref_sents)
        ):
            raise ValueError(
                f"Metric {metric_name} has different number of scored items than the number of source and reference "
                f"sentences! # src sents = {len(all_src_sents)}, # ref sents = {len(all_ref_sents)}, # "
                f"items in metric scores = {len(item_grouped_metric_scores)}."
            )
        if len(item_grouped_metric_scores) != len(item_grouped_gold_seg_scores):
            raise ValueError(
                f"Metric {metric_name} has different number of scored items than the gold {gold_name} scores! "
                f"# metric scored items = {len(item_grouped_metric_scores)}, # gold {gold_name} scored items "
                f"= {len(item_grouped_gold_seg_scores)}."
            )

        with open(
            lp_dir
            / METRIC_FILENAME_PATTERN.format(metric_name=metric_name, extension="txt"),
            "w",
        ) as metric_dump_file:
            for seg_idx, (
                src_sent,
                ref_sent,
                gold_item_scores,
                metric_item_scores,
            ) in enumerate(
                zip(
                    all_src_sents,
                    all_ref_sents,
                    item_grouped_gold_seg_scores,
                    item_grouped_metric_scores,
                )
            ):
                if len(gold_item_scores) == 0:
                    continue

                metric_valid_systems = valid_systems if not is_qe else valid_systems_qe
                if not (
                    len(metric_item_scores)
                    == len(metric_valid_systems)
                    == len(gold_item_scores)
                ):
                    raise ValueError(
                        f"Each item must have a number of scores equal to the number of the valid systems! "
                        f"# metric item scores = {len(metric_item_scores)}, # gold {gold_name} item scores = "
                        f"{len(gold_item_scores)}, # valid systems = {len(metric_valid_systems)}."
                    )

                for sys, gold_score, metric_score in zip(
                    metric_valid_systems, gold_item_scores, metric_item_scores
                ):
                    metric_dump_file.write(f"SRC: {src_sent}\n")
                    metric_dump_file.write(
                        f"{sys} CAND: {sys2all_cand_sents[sys][seg_idx]}\n"
                    )
                    metric_dump_file.write(f"REF: {ref_sent}\n")
                    metric_dump_file.write(f"GOLD {gold_name} score: {gold_score}\n")
                    metric_dump_file.write(f"{metric_name} score: {metric_score}\n")
                    metric_dump_file.write("\n\n\n")


if __name__ == "__main__":
    parser = read_arguments()
    args = parser.parse_args()

    testset = get_wmt_testset(args.testset_name, args.lp, True)
    (
        metric_name2item_grouped_seg_scores,
        valid_systems,
        valid_systems_qe,
    ) = get_grouped_metrics_scores(
        get_wmt_metric_name2scores(testset, args.ref_to_use, args.new_metrics),
        args.ref_to_use,
        args.include_human,
        args.include_outliers,
        args.include_ref_to_use,
        args.gold_name,
        "item",
        testset.DomainsPerSeg(),
        True,
        testset.human_sys_names,
        testset.outlier_sys_names,
    )
    save_metrics_dump(
        metric_name2item_grouped_seg_scores,
        testset,
        args.ref_to_use,
        args.gold_name,
        valid_systems,
        valid_systems_qe,
    )
