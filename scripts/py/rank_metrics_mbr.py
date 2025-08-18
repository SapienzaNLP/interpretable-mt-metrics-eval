import json
from argparse import ArgumentParser
from typing import Dict, List

from mt_metrics_thresholds.definitions import (
    METRICS_RANKINGS_DIR,
    SUBMITTED_METRICS_DIRNAME,
    NEW_METRICS_DIRNAME,
)
from mt_metrics_thresholds.utils.wmt import get_mbr_data


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to rank MT metrics on a WMT test set according to the MBR evaluation criterion."
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
        "--reverse-mbr",
        action="store_true",
        help="Whether to apply reverse MBR.",
    )
    return parser


def rank_metrics_for_mbr(
    metric_name2mbr_predictions: Dict[str, Dict[int, List[str]]],
    seg_idx2gold_sys_scores: Dict[int, Dict[str, float]],
    lp: str,
    new_metrics: bool,
    reverse_mbr: bool,
) -> None:
    """
    Rank MT metrics on a WMT23 test set according to the MBR evaluation criterion.

    :param metric_name2mbr_predictions: Dictionary from metric names to their MBR predictions.
    :param seg_idx2gold_sys_scores: Dictionary from segment indices to gold scores for each MT system.
    :param lp: Language pair to consider for WMT23.
    :param new_metrics: Whether the checkpoints of some metrics are the news ones (wrt wmt23).
    :param reverse_mbr: Whether to apply reverse MBR.
    """
    metric_name2mbr_eval_results = dict()
    rank_criterion = (
        "macro_mbr_precision" if not reverse_mbr else "macro_reverse_mbr_precision"
    )
    for metric_name, seg_idx2mbr_predictions in metric_name2mbr_predictions.items():
        tp, fp, fn, tn = 0, 0, 0, 0
        mbr_seg_precisions_sum = 0
        best_translation_metric_mqm_score_macro_avg = 0
        for seg_idx, mbr_predictions in seg_idx2mbr_predictions.items():
            sys2gold_score = seg_idx2gold_sys_scores[seg_idx]
            gt = []
            max_gold_score = float("-inf")
            for sys_name, gold_score in sys2gold_score.items():
                if gold_score == max_gold_score:
                    gt.append(sys_name)
                elif gold_score > max_gold_score:
                    max_gold_score = gold_score
                    gt = [sys_name]
            assert len(gt) > 0

            all_systems, gt, mbr_predictions = (
                set(sys2gold_score),
                set(gt),
                set(mbr_predictions),
            )
            seg_tp = len(gt & mbr_predictions)
            seg_fp = len(mbr_predictions - gt)
            seg_fn = len(gt - mbr_predictions)
            seg_tn = len(all_systems - (gt | mbr_predictions))
            seg_precision = seg_tp / (seg_tp + seg_fp)
            mbr_seg_precisions_sum += seg_precision
            tp += seg_tp
            fp += seg_fp
            fn += seg_fn
            tn += seg_tn

            best_translation_metric_mqm_score_macro_avg += sum(
                sys2gold_score[sys_name] for sys_name in mbr_predictions
            ) / len(mbr_predictions)

        macro_mbr_precision = mbr_seg_precisions_sum / len(seg_idx2mbr_predictions)
        overall_best_translation_metric_mqm_score_macro_avg = (
            best_translation_metric_mqm_score_macro_avg / len(seg_idx2mbr_predictions)
        )
        metric_name2mbr_eval_results[metric_name] = {
            rank_criterion: macro_mbr_precision,
            "overall_best_translation_metric_mqm_score_macro_avg": overall_best_translation_metric_mqm_score_macro_avg,
            "mbr_tp": tp,
            "mbr_fp": fp,
            "mbr_fn": fn,
            "mbr_tn": tn,
        }

    # Rank metrics according to the MBR evaluation criterion
    ranked_metrics = sorted(
        metric_name2mbr_eval_results.items(),
        key=lambda x: (
            x[1][rank_criterion],
            x[1]["overall_best_translation_metric_mqm_score_macro_avg"],
        ),
        reverse=True,
    )
    rankings_dir = (
        METRICS_RANKINGS_DIR
        / (SUBMITTED_METRICS_DIRNAME if not new_metrics else NEW_METRICS_DIRNAME)
        / "wmt23"
        / lp
        / rank_criterion
    )
    if not rankings_dir.exists():
        rankings_dir.mkdir(parents=True, exist_ok=True)
    with open(rankings_dir / "ranking.json", "w") as file:
        json.dump(dict(ranked_metrics), file, indent=4)


if __name__ == "__main__":
    parser = read_arguments()
    args = parser.parse_args()

    metric_name2mbr_predictions, seg_idx2gold_sys_scores = get_mbr_data(
        args.lp,
        args.ref_to_use,
        args.include_human,
        args.include_outliers,
        args.gold_name,
        args.new_metrics,
        args.reverse_mbr,
    )

    rank_metrics_for_mbr(
        metric_name2mbr_predictions,
        seg_idx2gold_sys_scores,
        args.lp,
        args.new_metrics,
        args.reverse_mbr,
    )
