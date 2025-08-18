import pickle
import statistics
from typing import Union, List, Literal, Optional, Tuple, Dict

import numpy as np

from mt_metrics_thresholds.definitions import (
    DISCRETE_METRICS,
    METRICS_FPS_DELTAS_DIR,
    SUBMITTED_METRICS_DIRNAME,
    NEW_METRICS_DIRNAME,
    GOLD_SCORE_THRESHOLD_DIRNAME_PATTERN,
    METRIC_FILENAME_PATTERN,
    METRICS_FPS_PLOTS_DIR,
    METRICS_PR_PLOTS_DIR,
    AVERAGE_BY_DIRNAME_PATTERN,
    METRICS_DUMPS,
    GOLD_SCORE_PRECISION_THRESHOLDS_DIRNAME_PATTERN,
)
from mt_metrics_thresholds.utils.plot import (
    save_data_distribution_plot,
    plot_precision_recall_curve,
    plot_interactive_heatmap,
    prepare_heatmap_data,
)


def get_precision_recall_fbeta(
    tp: int,
    fp: int,
    fn: int,
    beta: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F_beta given TP, FP, FN, and Beta value.

    :param tp: Number of True Positives.
    :param fp: Number of False Positives.
    :param fn: Number of False Negatives.
    :param beta: Beta value for F_beta computation. Default: 1.0.

    :return: Tuple of Precision, Recall, and F_Beta.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        f_beta = 0
    else:
        f_beta = (
            (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        )
    return precision, recall, f_beta


def get_percentile_cand_thresholds(
    cand_thresholds: List[float], percentile_ids: List[float]
) -> List[float]:
    """
    Get percentile candidate thresholds.

    :param cand_thresholds: List of unique candidate thresholds.
    :param percentile_ids: Percentages to take into account for the percentiles' computation.

    :return: List of percentile thresholds.
    """
    return np.percentile(cand_thresholds, percentile_ids).tolist()


def process_metric(
    metric_name: str,
    grouped_metric_seg_scores: Union[List[float], List[List[float]]],
    grouped_gold_seg_scores: Union[List[float], List[List[float]]],
    average_by: Literal["none", "item", "sys"],
    testset_name: str,
    lp: str,
    metric_valid_systems: List[str],
    gold_score_threshold: float,
    precision_threshold: float,
    beta: float,
    new_metrics: bool,
    save_files: bool,
    percentile_ids: Optional[List[float]] = None,
    percentile_ids_for_fps_delta_distribution: Optional[List[int]] = None,
    all_src_sents: Optional[List[str]] = None,
    all_ref_sents: Optional[List[str]] = None,
    sys2all_cand_sents: Optional[Dict[str, List[str]]] = None,
    lp_offsets: Optional[List[int]] = None,
    thresholds_to_use: Optional[List[float]] = None,
    return_results_with_all_thresholds_to_use: bool = False,
) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """
    Process the input metric and return the results.

    :param metric_name: Metric name.
    :param grouped_metric_seg_scores: Grouped segment scores given in output by the metric.
    :param grouped_gold_seg_scores: Gold grouped segment scores.
    :param average_by: What to average over when computing final scores. Allowed values: 'none', 'item', 'sys'.
    :param testset_name: Name of the WMT test set to use.
    :param lp: Language pair to consider.
    :param metric_valid_systems: Valid systems based on which items' scores are sorted.
    :param gold_score_threshold: Gold score threshold to consider in the optimization.
    :param precision_threshold: Precision threshold to consider in the optimization.
    :param beta: Beta value for F_beta computation.
    :param new_metrics: Whether the checkpoints of some metrics are the news ones (wrt wmt23).
    :param save_files: Whether to save files containing the results (plots, logs, result pickle dicts, etc.).
    :param percentile_ids: Percentile IDs to consider for the percentile thresholds. Default: None.
    :param percentile_ids_for_fps_delta_distribution: Percentiles for the FPs delta distribution plots. Default: None.
    :param all_src_sents: List of source sentences (used for logging). Default: None.
    :param all_ref_sents: List of reference sentences (used for logging). Default: None.
    :param sys2all_cand_sents: Candidate translations dictionary (used for logging). Default: None.
    :param lp_offsets: End indices of the data for each language pair. Default: None.
    :param thresholds_to_use: Thresholds to use for the metric (if passed). Default: None.
    :param return_results_with_all_thresholds_to_use: Return the results for each threshold to use. Default: False.

    :return: Metric name, binary classification optimization result, and QE ranking optimization result.
    """

    if average_by != "none" and average_by != "item" and average_by != "sys":
        raise ValueError(
            f"Invalid 'average_by' parameter: {average_by}! Allowed values: 'none', 'item', 'sys'."
        )
    if return_results_with_all_thresholds_to_use and average_by != "sys":
        raise ValueError(
            "If 'return_results_with_all_thresholds_to_use' is True, 'average_by' must be 'sys'!"
        )

    best_macro_f1 = 0 if not return_results_with_all_thresholds_to_use else []
    if not return_results_with_all_thresholds_to_use:
        (
            optimal_micro_f1,
            optimal_macro_p,
            optimal_micro_p,
            optimal_macro_r,
            optimal_micro_r,
            optimal_metric_threshold,
            optimal_tp,
            optimal_fp,
            optimal_fn,
            optimal_tn,
        ) = (0, 0, 0, 0, 0, None, 0, 0, 0, 0)

        optimal_n_p_groups, optimal_n_r_groups = 0, 0
    else:
        (
            optimal_micro_f1,
            optimal_macro_p,
            optimal_micro_p,
            optimal_macro_r,
            optimal_micro_r,
            optimal_metric_threshold,
            optimal_tp,
            optimal_fp,
            optimal_fn,
            optimal_tn,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        optimal_n_p_groups, optimal_n_r_groups = [], []

    (
        optimal_fps_deltas,
        optimal_lp_fps_deltas,
        optimal_fps_deltas_avg,
        optimal_lp_fps_deltas_avg,
    ) = (
        None,
        None,
        None,
        None,
    )

    (
        optimal_tp_logs,
        optimal_fp_logs,
        optimal_fn_logs,
        optimal_tn_logs,
    ) = (
        [],
        [],
        [],
        [],
    )  # Used only in item-grouping and if all_src_sents is not None (logging)

    # Candidate thresholds are all distinct scores for the metric
    if thresholds_to_use is not None:
        cand_thresholds = sorted(thresholds_to_use)
    elif average_by == "none":
        cand_thresholds = sorted(set(grouped_metric_seg_scores))
    else:
        cand_thresholds = []
        for group_scores in grouped_metric_seg_scores:
            cand_thresholds += group_scores
        cand_thresholds = sorted(set(cand_thresholds))

    optimization_result, optimization_result_for_qe_ranking = None, None

    # QE statistics (used only in item-grouping)
    qe_ranking_tp, qe_ranking_fp, qe_ranking_fn, qe_ranking_tn = 0, 0, 0, 0
    n_qe_ranking_items, lp_n_qe_ranking_items = 0, 0
    lp_qe_ranking_precisions, lp_qe_ranking_recalls, lp_qe_ranking_f1_scores = (
        [],
        [],
        [],
    )
    qe_ranking_precisions_sum, qe_ranking_recalls_sum = 0, 0
    lp_qe_ranking_precisions_sum, lp_qe_ranking_recalls_sum = 0, 0

    qe_ranking_fps_deltas, qe_ranking_fps_deltas_avg = [], []
    lp_qe_ranking_fps_deltas = [[]]

    (
        lp_qe_ranking_best_translation_metric_mqm_score_macro_avg,
        lp_qe_ranking_best_translation_gold_mqm_score_macro_avg,
        lp_qe_ranking_best_translation_metric_mqm_score_micro_avg,
        lp_qe_ranking_best_translation_gold_mqm_score_micro_avg,
    ) = ([[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]])

    if thresholds_to_use is not None or percentile_ids is None:
        percentile_cand_thresholds = []
    elif metric_name not in DISCRETE_METRICS:
        percentile_cand_thresholds = get_percentile_cand_thresholds(
            cand_thresholds, percentile_ids
        )
    else:
        percentile_cand_thresholds = range(-25, 1)

    for thresholds_iteration, thresholds in enumerate(
        [cand_thresholds, percentile_cand_thresholds]
    ):
        macro_precisions, macro_recalls = [], []
        micro_precisions, micro_recalls = (
            [],
            [],
        )  # Macro and Micro is wrt the several lps (for when lp == "all")

        threshold2stats = dict()
        for threshold_id, threshold in enumerate(thresholds):
            tp, fp, fn, tn = 0, 0, 0, 0
            fps_deltas, fps_deltas_avg = [], []

            lp_idx = 0
            lp_precisions, lp_recalls, lp_f1_scores = [], [], []
            lp_fps_deltas = [[]]

            n_p_groups, n_r_groups = 0, 0  # Used only in item and sys grouping

            if average_by == "none":
                lp_tp, lp_fp, lp_fn = 0, 0, 0

                for i, (metric_score, gt) in enumerate(
                    zip(grouped_metric_seg_scores, grouped_gold_seg_scores)
                ):
                    if metric_score >= threshold and gt >= gold_score_threshold:
                        tp += 1
                        lp_tp += 1
                    elif metric_score >= threshold and gt < gold_score_threshold:
                        fp += 1
                        lp_fp += 1
                        fps_deltas.append(gt - gold_score_threshold)
                        if lp == "all":
                            lp_fps_deltas[-1].append(gt - gold_score_threshold)
                    elif metric_score < threshold and gt >= gold_score_threshold:
                        fn += 1
                        lp_fn += 1
                    else:
                        tn += 1

                    if lp == "all" and i == lp_offsets[lp_idx]:
                        precision, recall, f1 = get_precision_recall_fbeta(
                            lp_tp, lp_fp, lp_fn, beta
                        )
                        lp_precisions.append(precision)
                        lp_recalls.append(recall)
                        lp_f1_scores.append(f1)
                        lp_tp, lp_fp, lp_fn = 0, 0, 0

                        lp_fps_deltas.append(
                            []
                        )  # Maybe this addition of an empty list in the last iteration is wrong.

                        lp_idx += 1

                micro_precision, micro_recall, micro_f1 = get_precision_recall_fbeta(
                    tp, fp, fn, beta
                )
                if lp == "all":
                    macro_precision, macro_recall, macro_f1 = (
                        sum(lp_precisions) / len(lp_precisions),
                        sum(lp_recalls) / len(lp_recalls),
                        sum(lp_f1_scores) / len(lp_f1_scores),
                    )
                    micro_precisions.append(micro_precision)
                    micro_recalls.append(micro_recall)
                    macro_precisions.append(macro_precision)
                    macro_recalls.append(macro_recall)
                else:
                    macro_precision, macro_recall, macro_f1 = (
                        micro_precision,
                        micro_recall,
                        micro_f1,
                    )

                if (
                    thresholds_iteration == 0
                    and macro_precision >= precision_threshold
                    and macro_f1 > best_macro_f1
                ):
                    best_macro_f1 = macro_f1
                    (
                        optimal_micro_f1,
                        optimal_macro_p,
                        optimal_micro_p,
                        optimal_macro_r,
                        optimal_micro_r,
                        optimal_metric_threshold,
                        optimal_tp,
                        optimal_fp,
                        optimal_fn,
                        optimal_tn,
                    ) = (
                        micro_f1,
                        macro_precision,
                        micro_precision,
                        macro_recall,
                        micro_recall,
                        threshold,
                        tp,
                        fp,
                        fn,
                        tn,
                    )

                    optimal_fps_deltas = fps_deltas.copy()
                    optimal_fps_deltas_avg = fps_deltas.copy()
                    if lp == "all":
                        optimal_lp_fps_deltas = lp_fps_deltas.copy()

            else:
                tp_logs, fp_logs, fn_logs, tn_logs = [], [], [], []

                precisions_sum, recalls_sum = 0, 0

                lp_precisions_sum, lp_recalls_sum = 0, 0

                lp_n_p_groups, lp_n_r_groups = 0, 0

                for group_idx, (
                    metric_group_scores,
                    gold_group_scores,
                ) in enumerate(zip(grouped_metric_seg_scores, grouped_gold_seg_scores)):
                    if len(metric_group_scores) != len(gold_group_scores):
                        raise ValueError(
                            f"Metric {metric_name} has different number of scores than the gold scores for a group! "
                            f"# metric group scores = {len(metric_group_scores)}, # gold group scores = "
                            f"{len(gold_group_scores)}."
                        )

                    if (
                        len(gold_group_scores) > 0
                    ):  # There are some groups that do not have gold scores
                        group_tp, group_fp, group_fn, group_tn = 0, 0, 0, 0

                        metric_max_score, metric_max_score_ids = float("-inf"), []
                        gold_max_score, gold_max_score_ids = float("-inf"), []

                        group_fps_deltas_sum = 0

                        for elem_idx, (metric_score, gt) in enumerate(
                            zip(metric_group_scores, gold_group_scores)
                        ):  # elem_idx is the index that identifies the system only in item-grouping
                            if metric_score >= threshold and gt >= gold_score_threshold:
                                tp += 1
                                group_tp += 1
                                if all_src_sents is not None:
                                    tp_logs.append(
                                        (
                                            all_src_sents[group_idx],
                                            sys2all_cand_sents[
                                                metric_valid_systems[elem_idx]
                                            ][group_idx],
                                            all_ref_sents[group_idx],
                                            metric_score,
                                            gt,
                                        )
                                    )
                            elif (
                                metric_score >= threshold and gt < gold_score_threshold
                            ):
                                fp += 1
                                group_fp += 1
                                fps_deltas.append(gt - gold_score_threshold)
                                group_fps_deltas_sum += gt - gold_score_threshold
                                if lp == "all":
                                    lp_fps_deltas[-1].append(gt - gold_score_threshold)
                                if all_src_sents is not None:
                                    fp_logs.append(
                                        (
                                            all_src_sents[group_idx],
                                            sys2all_cand_sents[
                                                metric_valid_systems[elem_idx]
                                            ][group_idx],
                                            all_ref_sents[group_idx],
                                            metric_score,
                                            gt,
                                        )
                                    )
                            elif (
                                metric_score < threshold and gt >= gold_score_threshold
                            ):
                                fn += 1
                                group_fn += 1
                                if all_src_sents is not None:
                                    fn_logs.append(
                                        (
                                            all_src_sents[group_idx],
                                            sys2all_cand_sents[
                                                metric_valid_systems[elem_idx]
                                            ][group_idx],
                                            all_ref_sents[group_idx],
                                            metric_score,
                                            gt,
                                        )
                                    )
                            else:
                                tn += 1
                                if all_src_sents is not None:
                                    tn_logs.append(
                                        (
                                            all_src_sents[group_idx],
                                            sys2all_cand_sents[
                                                metric_valid_systems[elem_idx]
                                            ][group_idx],
                                            all_ref_sents[group_idx],
                                            metric_score,
                                            gt,
                                        )
                                    )

                            if (
                                average_by == "item"
                                and optimization_result_for_qe_ranking is None
                            ):
                                if metric_score > metric_max_score:
                                    (
                                        metric_max_score,
                                        metric_max_score_ids,
                                    ) = metric_score, [elem_idx]
                                elif metric_score == metric_max_score:
                                    metric_max_score_ids.append(elem_idx)

                                if gt > gold_max_score:
                                    gold_max_score, gold_max_score_ids = gt, [elem_idx]
                                elif gt == gold_max_score:
                                    gold_max_score_ids.append(elem_idx)

                        if group_fp > 0:
                            fps_deltas_avg.append(group_fps_deltas_sum / group_fp)

                        precision, recall, _ = get_precision_recall_fbeta(
                            group_tp, group_fp, group_fn, beta
                        )
                        if group_tp + group_fp > 0:
                            precisions_sum += precision
                            n_p_groups += 1

                            lp_precisions_sum += precision
                            lp_n_p_groups += 1
                        if group_tp + group_fn > 0:
                            recalls_sum += recall
                            n_r_groups += 1

                            lp_recalls_sum += recall
                            lp_n_r_groups += 1

                        if (
                            average_by == "item"
                            and optimization_result_for_qe_ranking is None
                        ):
                            best_predicted_candidate_ids, best_gt_candidate_ids = set(
                                metric_max_score_ids
                            ), set(gold_max_score_ids)

                            qe_ranking_item_tp = len(
                                best_predicted_candidate_ids & best_gt_candidate_ids
                            )
                            fp_ids = (
                                best_predicted_candidate_ids - best_gt_candidate_ids
                            )
                            qe_ranking_item_fp = len(fp_ids)
                            qe_ranking_item_fn = len(
                                best_gt_candidate_ids - best_predicted_candidate_ids
                            )
                            qe_ranking_item_tn = len(gold_group_scores) - len(
                                best_predicted_candidate_ids | best_gt_candidate_ids
                            )
                            qe_ranking_tp += qe_ranking_item_tp
                            qe_ranking_fp += qe_ranking_item_fp
                            qe_ranking_fn += qe_ranking_item_fn
                            qe_ranking_tn += qe_ranking_item_tn

                            precision, recall, _ = get_precision_recall_fbeta(
                                qe_ranking_item_tp,
                                qe_ranking_item_fp,
                                qe_ranking_item_fn,
                                beta,
                            )

                            qe_ranking_precisions_sum += precision
                            qe_ranking_recalls_sum += recall
                            n_qe_ranking_items += 1

                            lp_qe_ranking_precisions_sum += precision
                            lp_qe_ranking_recalls_sum += recall
                            lp_n_qe_ranking_items += 1

                            seg_qe_ranking_fps_sum = 0
                            for fp_idx in fp_ids:
                                qe_ranking_fp_delta = (
                                    gold_group_scores[fp_idx] - gold_max_score
                                )
                                assert qe_ranking_fp_delta < 0
                                qe_ranking_fps_deltas.append(qe_ranking_fp_delta)
                                seg_qe_ranking_fps_sum += qe_ranking_fp_delta
                                if lp == "all":
                                    lp_qe_ranking_fps_deltas[-1].append(
                                        qe_ranking_fp_delta
                                    )
                            if len(fp_ids) > 0:
                                qe_ranking_fps_deltas_avg.append(
                                    seg_qe_ranking_fps_sum / len(fp_ids)
                                )

                            metric_mqm_scores, gold_mqm_scores = [
                                gold_group_scores[score_id]
                                for score_id in metric_max_score_ids
                            ], [
                                gold_group_scores[score_id]
                                for score_id in gold_max_score_ids
                            ]

                            metric_mqm_scores_sum, gold_mqm_scores_sum = sum(
                                metric_mqm_scores
                            ), sum(gold_mqm_scores)

                            lp_qe_ranking_best_translation_metric_mqm_score_macro_avg[
                                -1
                            ][0] += metric_mqm_scores_sum / len(metric_mqm_scores)
                            lp_qe_ranking_best_translation_metric_mqm_score_macro_avg[
                                -1
                            ][1] += 1
                            lp_qe_ranking_best_translation_gold_mqm_score_macro_avg[-1][
                                0
                            ] += gold_mqm_scores_sum / len(gold_mqm_scores)
                            lp_qe_ranking_best_translation_gold_mqm_score_macro_avg[-1][
                                1
                            ] += 1

                            lp_qe_ranking_best_translation_metric_mqm_score_micro_avg[
                                -1
                            ][0] += metric_mqm_scores_sum
                            lp_qe_ranking_best_translation_metric_mqm_score_micro_avg[
                                -1
                            ][1] += len(metric_mqm_scores)
                            lp_qe_ranking_best_translation_gold_mqm_score_micro_avg[-1][
                                0
                            ] += gold_mqm_scores_sum
                            lp_qe_ranking_best_translation_gold_mqm_score_micro_avg[-1][
                                1
                            ] += len(gold_mqm_scores)

                    if lp == "all" and group_idx == lp_offsets[lp_idx]:
                        lp_precision, lp_recall = None, None

                        if (
                            lp_n_p_groups > 0
                        ):  # With all lps, some thresholds may not have supports for some lps
                            lp_precision = lp_precisions_sum / lp_n_p_groups
                            lp_precisions.append(lp_precision)
                        lp_precisions_sum, lp_n_p_groups = 0, 0

                        if lp_n_r_groups > 0:
                            lp_recall = lp_recalls_sum / lp_n_r_groups
                            lp_recalls.append(lp_recall)
                        lp_recalls_sum, lp_n_r_groups = 0, 0

                        if lp_precision is not None and lp_recall is not None:
                            lp_f1_scores.append(
                                (1 + beta**2)
                                * (lp_precision * lp_recall)
                                / (beta**2 * lp_precision + lp_recall)
                                if lp_precision + lp_recall > 0
                                else 0
                            )

                        lp_fps_deltas.append([])

                        if (
                            average_by == "item"
                            and optimization_result_for_qe_ranking is None
                        ):
                            lp_qe_ranking_precision = (
                                lp_qe_ranking_precisions_sum / lp_n_qe_ranking_items
                            )
                            lp_qe_ranking_precisions.append(lp_qe_ranking_precision)
                            lp_qe_ranking_precisions_sum, lp_n_qe_ranking_items = 0, 0

                            lp_qe_ranking_recall = (
                                lp_qe_ranking_recalls_sum / lp_n_qe_ranking_items
                            )
                            lp_qe_ranking_recalls.append(lp_qe_ranking_recall)
                            lp_qe_ranking_recalls_sum, lp_n_qe_ranking_items = 0, 0

                            lp_qe_ranking_f1_scores.append(
                                (1 + beta**2)
                                * (lp_qe_ranking_precision * lp_qe_ranking_recall)
                                / (
                                    beta**2 * lp_qe_ranking_precision
                                    + lp_qe_ranking_recall
                                )
                                if lp_qe_ranking_precision + lp_qe_ranking_recall > 0
                                else 0
                            )

                            lp_qe_ranking_fps_deltas.append([])

                            lp_qe_ranking_best_translation_metric_mqm_score_macro_avg.append(
                                [0, 0]
                            )
                            lp_qe_ranking_best_translation_gold_mqm_score_macro_avg.append(
                                [0, 0]
                            )
                            lp_qe_ranking_best_translation_metric_mqm_score_micro_avg.append(
                                [0, 0]
                            )
                            lp_qe_ranking_best_translation_gold_mqm_score_micro_avg.append(
                                [0, 0]
                            )

                        lp_idx += 1

                micro_precision, micro_recall = (
                    precisions_sum / n_p_groups,
                    recalls_sum / n_r_groups,
                )
                micro_f1 = (
                    (1 + beta**2)
                    * (micro_precision * micro_recall)
                    / (beta**2 * micro_precision + micro_recall)
                    if micro_precision + micro_recall > 0
                    else 0
                )
                (
                    micro_qe_ranking_precision,
                    micro_qe_ranking_recall,
                    micro_qe_ranking_f1,
                ) = (None, None, None)
                if average_by == "item" and optimization_result_for_qe_ranking is None:
                    micro_qe_ranking_precision, micro_qe_ranking_recall = (
                        qe_ranking_precisions_sum / n_qe_ranking_items,
                        qe_ranking_recalls_sum / n_qe_ranking_items,
                    )
                    micro_qe_ranking_f1 = (
                        (1 + beta**2)
                        * (micro_qe_ranking_precision * micro_qe_ranking_recall)
                        / (
                            beta**2 * micro_qe_ranking_precision
                            + micro_qe_ranking_recall
                        )
                        if micro_qe_ranking_precision + micro_qe_ranking_recall > 0
                        else 0
                    )

                if lp == "all":
                    macro_precision, macro_recall, macro_f1 = (
                        sum(lp_precisions) / len(lp_precisions),
                        sum(lp_recalls) / len(lp_recalls),
                        sum(lp_f1_scores) / len(lp_f1_scores),
                    )
                    macro_precisions.append(macro_precision)
                    macro_recalls.append(macro_recall)

                    if (
                        average_by == "item"
                        and optimization_result_for_qe_ranking is None
                    ):
                        (
                            macro_qe_ranking_precision,
                            macro_qe_ranking_recall,
                            macro_qe_ranking_f1,
                        ) = (
                            sum(lp_qe_ranking_precisions)
                            / len(lp_qe_ranking_precisions),
                            sum(lp_qe_ranking_recalls) / len(lp_qe_ranking_recalls),
                            sum(lp_qe_ranking_f1_scores) / len(lp_qe_ranking_f1_scores),
                        )
                    else:
                        (
                            macro_qe_ranking_precision,
                            macro_qe_ranking_recall,
                            macro_qe_ranking_f1,
                        ) = (None, None, None)

                else:
                    macro_precision, macro_recall, macro_f1 = (
                        micro_precision,
                        micro_recall,
                        micro_f1,
                    )

                    (
                        macro_qe_ranking_precision,
                        macro_qe_ranking_recall,
                        macro_qe_ranking_f1,
                    ) = (
                        micro_qe_ranking_precision,
                        micro_qe_ranking_recall,
                        micro_qe_ranking_f1,
                    )

                micro_precisions.append(micro_precision)
                micro_recalls.append(micro_recall)

                if thresholds_iteration == 0 and (
                    return_results_with_all_thresholds_to_use
                    or (
                        (
                            macro_precision >= precision_threshold
                            and macro_f1 > best_macro_f1
                        )
                    )
                ):
                    if not return_results_with_all_thresholds_to_use:
                        best_macro_f1 = macro_f1
                        (
                            optimal_micro_f1,
                            optimal_macro_p,
                            optimal_micro_p,
                            optimal_macro_r,
                            optimal_micro_r,
                            optimal_metric_threshold,
                            optimal_tp,
                            optimal_fp,
                            optimal_fn,
                            optimal_tn,
                        ) = (
                            micro_f1,
                            macro_precision,
                            micro_precision,
                            macro_recall,
                            micro_recall,
                            threshold,
                            tp,
                            fp,
                            fn,
                            tn,
                        )

                        optimal_fps_deltas = fps_deltas.copy()
                        optimal_fps_deltas_avg = fps_deltas_avg.copy()
                        if lp == "all":
                            optimal_lp_fps_deltas = lp_fps_deltas.copy()

                        optimal_n_p_groups, optimal_n_r_groups = n_p_groups, n_r_groups

                        (
                            optimal_tp_logs,
                            optimal_fp_logs,
                            optimal_fn_logs,
                            optimal_tn_logs,
                        ) = (tp_logs, fp_logs, fn_logs, tn_logs)
                    else:
                        best_macro_f1.append(macro_f1)
                        optimal_micro_f1.append(micro_f1)
                        optimal_macro_p.append(macro_precision)
                        optimal_micro_p.append(micro_precision)
                        optimal_macro_r.append(macro_recall)
                        optimal_micro_r.append(micro_recall)
                        optimal_metric_threshold.append(threshold)
                        optimal_tp.append(tp)
                        optimal_fp.append(fp)
                        optimal_fn.append(fn)
                        optimal_tn.append(tn)

                        optimal_n_p_groups.append(n_p_groups)
                        optimal_n_r_groups.append(n_r_groups)

                if optimization_result_for_qe_ranking is None and average_by == "item":
                    overall_best_translation_metric_mqm_score_macro_avg = sum(
                        score_sum
                        for score_sum, n_segs in lp_qe_ranking_best_translation_metric_mqm_score_macro_avg
                    ) / sum(
                        n_segs
                        for score_sum, n_segs in lp_qe_ranking_best_translation_metric_mqm_score_macro_avg
                    )
                    mean_best_translation_metric_mqm_score_macro_avg = sum(
                        score_sum / n_segs
                        for score_sum, n_segs in lp_qe_ranking_best_translation_metric_mqm_score_macro_avg
                    ) / len(lp_qe_ranking_best_translation_metric_mqm_score_macro_avg)

                    overall_best_translation_gold_mqm_score_macro_avg = sum(
                        score_sum
                        for score_sum, n_segs in lp_qe_ranking_best_translation_gold_mqm_score_macro_avg
                    ) / sum(
                        n_segs
                        for score_sum, n_segs in lp_qe_ranking_best_translation_gold_mqm_score_macro_avg
                    )
                    mean_best_translation_gold_mqm_score_macro_avg = sum(
                        score_sum / n_segs
                        for score_sum, n_segs in lp_qe_ranking_best_translation_gold_mqm_score_macro_avg
                    ) / len(lp_qe_ranking_best_translation_gold_mqm_score_macro_avg)

                    overall_best_translation_metric_mqm_score_micro_avg = sum(
                        score_sum
                        for score_sum, n_scores in lp_qe_ranking_best_translation_metric_mqm_score_micro_avg
                    ) / sum(
                        n_scores
                        for score_sum, n_scores in lp_qe_ranking_best_translation_metric_mqm_score_micro_avg
                    )
                    mean_best_translation_metric_mqm_score_micro_avg = sum(
                        score_sum / n_scores
                        for score_sum, n_scores in lp_qe_ranking_best_translation_metric_mqm_score_micro_avg
                    ) / len(lp_qe_ranking_best_translation_metric_mqm_score_micro_avg)

                    overall_best_translation_gold_mqm_score_micro_avg = sum(
                        score_sum
                        for score_sum, n_scores in lp_qe_ranking_best_translation_gold_mqm_score_micro_avg
                    ) / sum(
                        n_scores
                        for score_sum, n_scores in lp_qe_ranking_best_translation_gold_mqm_score_micro_avg
                    )
                    mean_best_translation_gold_mqm_score_micro_avg = sum(
                        score_sum / n_scores
                        for score_sum, n_scores in lp_qe_ranking_best_translation_gold_mqm_score_micro_avg
                    ) / len(lp_qe_ranking_best_translation_gold_mqm_score_micro_avg)

                    optimization_result_for_qe_ranking = {
                        "macro_qe_ranking_precision": macro_qe_ranking_precision,
                        "micro_qe_ranking_precision": micro_qe_ranking_precision,
                        "macro_qe_ranking_recall": macro_qe_ranking_recall,
                        "micro_qe_ranking_recall": micro_qe_ranking_recall,
                        "macro_qe_ranking_f1": macro_qe_ranking_f1,
                        "micro_qe_ranking_f1": micro_qe_ranking_f1,
                        "qe_ranking_tp": qe_ranking_tp,
                        "qe_ranking_fp": qe_ranking_fp,
                        "qe_ranking_fn": qe_ranking_fn,
                        "qe_ranking_tn": qe_ranking_tn,
                        "overall_best_translation_metric_mqm_score_macro_avg": overall_best_translation_metric_mqm_score_macro_avg,
                        "mean_best_translation_metric_mqm_score_macro_avg": mean_best_translation_metric_mqm_score_macro_avg,
                        "overall_best_translation_gold_mqm_score_macro_avg": overall_best_translation_gold_mqm_score_macro_avg,
                        "mean_best_translation_gold_mqm_score_macro_avg": mean_best_translation_gold_mqm_score_macro_avg,
                        "overall_best_translation_metric_mqm_score_micro_avg": overall_best_translation_metric_mqm_score_micro_avg,
                        "mean_best_translation_metric_mqm_score_micro_avg": mean_best_translation_metric_mqm_score_micro_avg,
                        "overall_best_translation_gold_mqm_score_micro_avg": overall_best_translation_gold_mqm_score_micro_avg,
                        "mean_best_translation_gold_mqm_score_micro_avg": mean_best_translation_gold_mqm_score_micro_avg,
                    }
                    if lp == "all":
                        (
                            lp_qe_ranking_fps_deltas,
                            lp_median_fps_deltas,
                            lp_std_deviation_fps_delta,
                        ) = (
                            [
                                statistics.mean(lp_fps) if len(lp_fps) > 0 else 0
                                for lp_fps in lp_qe_ranking_fps_deltas
                            ],
                            [
                                statistics.median(lp_fps) if len(lp_fps) > 0 else 0
                                for lp_fps in lp_qe_ranking_fps_deltas
                            ],
                            [
                                statistics.stdev(lp_fps) if len(lp_fps) > 1 else 0
                                for lp_fps in lp_qe_ranking_fps_deltas
                            ],
                        )

                        optimization_result_for_qe_ranking[
                            "micro_mean_qe_ranking_fps_delta"
                        ] = (
                            statistics.mean(qe_ranking_fps_deltas)
                            if len(qe_ranking_fps_deltas) > 0
                            else 0
                        )
                        optimization_result_for_qe_ranking[
                            "macro_mean_qe_ranking_fps_delta"
                        ] = sum(lp_qe_ranking_fps_deltas) / len(
                            lp_qe_ranking_fps_deltas
                        )
                        optimization_result_for_qe_ranking[
                            "micro_median_qe_ranking_fps_delta"
                        ] = (
                            statistics.median(qe_ranking_fps_deltas)
                            if len(qe_ranking_fps_deltas) > 0
                            else 0
                        )
                        optimization_result_for_qe_ranking[
                            "macro_median_qe_ranking_fps_delta"
                        ] = sum(lp_median_fps_deltas) / len(lp_median_fps_deltas)
                        optimization_result_for_qe_ranking[
                            "micro_std_deviation_qe_ranking_fps_delta"
                        ] = (
                            statistics.stdev(qe_ranking_fps_deltas)
                            if len(qe_ranking_fps_deltas) > 1
                            else 0
                        )
                        optimization_result_for_qe_ranking[
                            "macro_std_deviation_qe_ranking_fps_delta"
                        ] = sum(lp_std_deviation_fps_delta) / len(
                            lp_std_deviation_fps_delta
                        )
                    else:
                        optimization_result_for_qe_ranking[
                            "micro_mean_qe_ranking_fps_delta"
                        ] = (
                            statistics.mean(qe_ranking_fps_deltas)
                            if len(qe_ranking_fps_deltas) > 0
                            else 0
                        )
                        optimization_result_for_qe_ranking[
                            "macro_mean_qe_ranking_fps_delta"
                        ] = (
                            statistics.mean(qe_ranking_fps_deltas_avg)
                            if len(qe_ranking_fps_deltas_avg) > 0
                            else 0
                        )
                        optimization_result_for_qe_ranking[
                            "micro_median_qe_ranking_fps_delta"
                        ] = (
                            statistics.median(qe_ranking_fps_deltas)
                            if len(qe_ranking_fps_deltas) > 0
                            else 0
                        )
                        optimization_result_for_qe_ranking[
                            "macro_median_qe_ranking_fps_delta"
                        ] = (
                            statistics.median(qe_ranking_fps_deltas_avg)
                            if len(qe_ranking_fps_deltas_avg) > 0
                            else 0
                        )
                        optimization_result_for_qe_ranking[
                            "micro_std_deviation_qe_ranking_fps_delta"
                        ] = (
                            statistics.stdev(qe_ranking_fps_deltas)
                            if len(qe_ranking_fps_deltas) > 1
                            else 0
                        )
                        optimization_result_for_qe_ranking[
                            "macro_std_deviation_qe_ranking_fps_delta"
                        ] = (
                            statistics.stdev(qe_ranking_fps_deltas_avg)
                            if len(qe_ranking_fps_deltas_avg) > 1
                            else 0
                        )

                    if save_files:
                        qe_ranking_fps_deltas_dir = (
                            METRICS_FPS_DELTAS_DIR
                            / (
                                SUBMITTED_METRICS_DIRNAME
                                if not new_metrics
                                else NEW_METRICS_DIRNAME
                            )
                            / testset_name
                            / lp
                            / GOLD_SCORE_THRESHOLD_DIRNAME_PATTERN.format(
                                gold_score_threshold=gold_score_threshold
                            )
                            / "qe_ranking"
                            / "micro"
                        )
                        if not qe_ranking_fps_deltas_dir.exists():
                            qe_ranking_fps_deltas_dir.mkdir(parents=True, exist_ok=True)
                        with open(
                            qe_ranking_fps_deltas_dir
                            / METRIC_FILENAME_PATTERN.format(
                                metric_name=metric_name, extension="pickle"
                            ),
                            "wb",
                        ) as f:
                            pickle.dump(
                                qe_ranking_fps_deltas,
                                f,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            )

                        macro_fps_deltas_dir = (
                            qe_ranking_fps_deltas_dir.parent / "macro"
                        )
                        if not macro_fps_deltas_dir.exists():
                            macro_fps_deltas_dir.mkdir(parents=True, exist_ok=True)
                        with open(
                            macro_fps_deltas_dir
                            / METRIC_FILENAME_PATTERN.format(
                                metric_name=metric_name, extension="pickle"
                            ),
                            "wb",
                        ) as f:
                            pickle.dump(
                                qe_ranking_fps_deltas_avg,
                                f,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            )

            if thresholds_iteration == 1 and thresholds_to_use is None and save_files:
                if lp == "all":
                    (
                        lp_fps_deltas,
                        lp_median_fps_deltas,
                        lp_std_deviation_fps_delta,
                    ) = (
                        [
                            statistics.mean(lp_fps) if len(lp_fps) > 0 else 0
                            for lp_fps in lp_fps_deltas
                        ],
                        [
                            statistics.median(lp_fps) if len(lp_fps) > 0 else 0
                            for lp_fps in lp_fps_deltas
                        ],
                        [
                            statistics.stdev(lp_fps) if len(lp_fps) > 1 else 0
                            for lp_fps in lp_fps_deltas
                        ],
                    )
                    threshold2stats[threshold] = {
                        "micro_mean_fps_delta": statistics.mean(fps_deltas)
                        if len(fps_deltas) > 0
                        else 0,
                        "macro_mean_fps_delta": sum(lp_fps_deltas) / len(lp_fps_deltas),
                        "micro_median_fps_delta": statistics.median(fps_deltas)
                        if len(fps_deltas) > 0
                        else 0,
                        "macro_median_fps_delta": sum(lp_median_fps_deltas)
                        / len(lp_median_fps_deltas),
                        "micro_std_deviation_fps_delta": statistics.stdev(fps_deltas)
                        if len(fps_deltas) > 1
                        else 0,
                        "macro_std_deviation_fps_delta": sum(lp_std_deviation_fps_delta)
                        / len(lp_std_deviation_fps_delta),
                        "micro_precision": micro_precision,
                        "micro_recall": micro_recall,
                        "micro_f1": micro_f1,
                        "macro_precision": macro_precision,
                        "macro_recall": macro_recall,
                        "macro_f1": macro_f1,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }
                else:
                    threshold2stats[threshold] = {
                        "mean_fps_delta": statistics.mean(fps_deltas)
                        if len(fps_deltas) > 0
                        else 0,
                        "median_fps_delta": statistics.median(fps_deltas)
                        if len(fps_deltas) > 0
                        else 0,
                        "std_deviation_fps_delta": statistics.stdev(fps_deltas)
                        if len(fps_deltas) > 1
                        else 0,
                        "precision": micro_precision,
                        "recall": micro_recall,
                        "f1": micro_f1,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }
                if average_by == "item":
                    (
                        threshold2stats[threshold][f"n_precision_items"],
                        threshold2stats[threshold][f"n_recall_items"],
                    ) = (n_p_groups, n_r_groups)
                elif average_by == "sys":
                    (
                        threshold2stats[threshold][f"n_precision_systems"],
                        threshold2stats[threshold][f"n_recall_systems"],
                    ) = (n_p_groups, n_r_groups)

                if threshold_id % 5 == 0:
                    percentile_dir = (
                        METRICS_FPS_PLOTS_DIR
                        / (
                            SUBMITTED_METRICS_DIRNAME
                            if not new_metrics
                            else NEW_METRICS_DIRNAME
                        )
                        / testset_name
                        / lp
                        / GOLD_SCORE_THRESHOLD_DIRNAME_PATTERN.format(
                            gold_score_threshold=gold_score_threshold
                        )
                        / "delta_distributions_on_percentiles"
                        / f"{percentile_ids_for_fps_delta_distribution[threshold_id // 5]}"
                    )
                    if not percentile_dir.exists():
                        percentile_dir.mkdir(parents=True, exist_ok=True)
                    save_data_distribution_plot(
                        fps_deltas,
                        "auto",
                        f"Distribution of FPs Deltas for {metric_name}",
                        "FP Delta",
                        "Density",
                        percentile_dir
                        / METRIC_FILENAME_PATTERN.format(
                            metric_name=metric_name, extension="png"
                        ),
                    )

        if thresholds_iteration == 0:
            save_dir = (
                METRICS_PR_PLOTS_DIR
                / (
                    SUBMITTED_METRICS_DIRNAME
                    if not new_metrics
                    else NEW_METRICS_DIRNAME
                )
                / testset_name
                / lp
                / GOLD_SCORE_THRESHOLD_DIRNAME_PATTERN.format(
                    gold_score_threshold=gold_score_threshold
                )
                / AVERAGE_BY_DIRNAME_PATTERN.format(average_by=average_by)
            )
            macro_ap, macro_auc_pr, macro_auc_pr_no_artificial_endpoint = (
                None,
                None,
                None,
            )
            if lp == "all":
                save_dir /= "macro"
                if not save_dir.exists():
                    save_dir.mkdir(parents=True, exist_ok=True)

                (
                    macro_ap,
                    macro_auc_pr,
                    macro_auc_pr_no_artificial_endpoint,
                ) = plot_precision_recall_curve(
                    metric_name,
                    average_by,
                    save_dir
                    / METRIC_FILENAME_PATTERN.format(
                        metric_name=metric_name, extension="png"
                    ),
                    precisions=macro_precisions,
                    recalls=macro_recalls,
                )

                save_dir = save_dir.parent / "micro"

            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)

            (
                micro_ap,
                micro_auc_pr,
                micro_auc_pr_no_artificial_endpoint,
            ) = (
                plot_precision_recall_curve(
                    metric_name,
                    average_by,
                    save_dir
                    / METRIC_FILENAME_PATTERN.format(
                        metric_name=metric_name, extension="png"
                    ),
                    grouped_metric_seg_scores if average_by == "none" else None,
                    [int(gt >= gold_score_threshold) for gt in grouped_gold_seg_scores]
                    if average_by == "none"
                    else None,
                    micro_precisions if average_by != "none" else None,
                    micro_recalls if average_by != "none" else None,
                )
                if thresholds_to_use is None and save_files
                else (None, None, None)
            )
            if lp != "all":
                macro_ap, macro_auc_pr, macro_auc_pr_no_artificial_endpoint = (
                    micro_ap,
                    micro_auc_pr,
                    micro_auc_pr_no_artificial_endpoint,
                )

            if all_src_sents is not None and save_files:
                save_dir = (
                    METRICS_DUMPS
                    / (
                        SUBMITTED_METRICS_DIRNAME
                        if not new_metrics
                        else NEW_METRICS_DIRNAME
                    )
                    / testset_name
                    / lp
                    / "optim_logs"
                    / GOLD_SCORE_PRECISION_THRESHOLDS_DIRNAME_PATTERN.format(
                        gold_score_threshold=gold_score_threshold,
                        precision_threshold=precision_threshold,
                    )
                    / metric_name
                )
                if not save_dir.exists():
                    save_dir.mkdir(parents=True, exist_ok=True)
                for log_file, log_triples in zip(
                    ["tps.txt", "fps.txt", "fns.txt", "tns.txt"],
                    [
                        optimal_tp_logs,
                        optimal_fp_logs,
                        optimal_fn_logs,
                        optimal_tn_logs,
                    ],
                ):
                    with open(save_dir / log_file, "w") as f:
                        for src, cand, ref, metric_score, gt in log_triples:
                            f.write(
                                f"SRC: {src}\nCAND: {cand}\nREF: {ref}\nMETRIC SCORE: {metric_score}\nHUMAN SCORE: "
                                f"{gt}\n\n\n"
                            )

            (
                optimal_micro_mean_fps_delta,
                optimal_macro_mean_fps_delta,
                optimal_micro_median_fps_delta,
                optimal_macro_median_fps_delta,
                optimal_micro_std_deviation_fps_delta,
                optimal_macro_std_deviation_fps_delta,
            ) = (None, None, None, None, None, None)
            if lp == "all":
                (
                    lp_fps_deltas,
                    lp_median_fps_deltas,
                    lp_std_deviation_fps_delta,
                ) = (
                    [
                        statistics.mean(lp_fps) if len(lp_fps) > 0 else 0
                        for lp_fps in optimal_lp_fps_deltas
                    ],
                    [
                        statistics.median(lp_fps) if len(lp_fps) > 0 else 0
                        for lp_fps in optimal_lp_fps_deltas
                    ],
                    [
                        statistics.stdev(lp_fps) if len(lp_fps) > 1 else 0
                        for lp_fps in optimal_lp_fps_deltas
                    ],
                )
                optimal_micro_mean_fps_delta = (
                    statistics.mean(optimal_fps_deltas)
                    if len(optimal_fps_deltas) > 0
                    else 0
                )
                optimal_macro_mean_fps_delta = sum(lp_fps_deltas) / len(lp_fps_deltas)
                optimal_micro_median_fps_delta = (
                    statistics.median(optimal_fps_deltas)
                    if len(optimal_fps_deltas) > 0
                    else 0
                )
                optimal_macro_median_fps_delta = sum(lp_median_fps_deltas) / len(
                    lp_median_fps_deltas
                )
                optimal_micro_std_deviation_fps_delta = (
                    statistics.stdev(optimal_fps_deltas)
                    if len(optimal_fps_deltas) > 1
                    else 0
                )
                optimal_macro_std_deviation_fps_delta = sum(
                    lp_std_deviation_fps_delta
                ) / len(lp_std_deviation_fps_delta)

            elif not return_results_with_all_thresholds_to_use:
                optimal_micro_mean_fps_delta = (
                    statistics.mean(optimal_fps_deltas)
                    if len(optimal_fps_deltas) > 0
                    else 0
                )
                optimal_macro_mean_fps_delta = (
                    statistics.mean(optimal_fps_deltas_avg)
                    if len(optimal_fps_deltas) > 0
                    else 0
                )
                optimal_micro_median_fps_delta = (
                    statistics.median(optimal_fps_deltas)
                    if len(optimal_fps_deltas) > 0
                    else 0
                )
                optimal_macro_median_fps_delta = (
                    statistics.median(optimal_fps_deltas_avg)
                    if len(optimal_fps_deltas) > 0
                    else 0
                )
                optimal_micro_std_deviation_fps_delta = (
                    statistics.stdev(optimal_fps_deltas)
                    if len(optimal_fps_deltas) > 1
                    else 0
                )
                optimal_macro_std_deviation_fps_delta = (
                    statistics.stdev(optimal_fps_deltas_avg)
                    if len(optimal_fps_deltas) > 1
                    else 0
                )

            optimization_result = {
                "optimal_metric_threshold": optimal_metric_threshold,
                "macro_f1": best_macro_f1,
                "micro_f1": optimal_micro_f1,
                "macro_precision": optimal_macro_p,
                "micro_precision": optimal_micro_p,
                "macro_recall": optimal_macro_r,
                "micro_recall": optimal_micro_r,
                "tp": optimal_tp,
                "fp": optimal_fp,
                "fn": optimal_fn,
                "tn": optimal_tn,
                "optimal_micro_mean_fps_delta": optimal_micro_mean_fps_delta,
                "optimal_macro_mean_fps_delta": optimal_macro_mean_fps_delta,
                "optimal_micro_median_fps_delta": optimal_micro_median_fps_delta,
                "optimal_macro_median_fps_delta": optimal_macro_median_fps_delta,
                "optimal_micro_std_deviation_fps_delta": optimal_micro_std_deviation_fps_delta,
                "optimal_macro_std_deviation_fps_delta": optimal_macro_std_deviation_fps_delta,
                "macro_ap": macro_ap,
                "macro_auc_pr": macro_auc_pr,
                "macro_auc_pr_no_artificial_endpoint": macro_auc_pr_no_artificial_endpoint,
                "micro_ap": micro_ap,
                "micro_auc_pr": micro_auc_pr,
                "micro_auc_pr_no_artificial_endpoint": micro_auc_pr_no_artificial_endpoint,
            }

            if average_by == "item":
                (
                    optimization_result["n_precision_items"],
                    optimization_result["n_recall_items"],
                ) = (optimal_n_p_groups, optimal_n_r_groups)
            elif average_by == "sys":
                (
                    optimization_result["n_precision_systems"],
                    optimization_result["n_recall_systems"],
                ) = (optimal_n_p_groups, optimal_n_r_groups)

            if save_files and not return_results_with_all_thresholds_to_use:
                fps_deltas_dir = (
                    METRICS_FPS_DELTAS_DIR
                    / (
                        SUBMITTED_METRICS_DIRNAME
                        if not new_metrics
                        else NEW_METRICS_DIRNAME
                    )
                    / testset_name
                    / lp
                    / GOLD_SCORE_THRESHOLD_DIRNAME_PATTERN.format(
                        gold_score_threshold=gold_score_threshold
                    )
                    / AVERAGE_BY_DIRNAME_PATTERN.format(average_by=average_by)
                    / "micro"
                )

                if not fps_deltas_dir.exists():
                    fps_deltas_dir.mkdir(parents=True, exist_ok=True)

                with open(
                    fps_deltas_dir
                    / METRIC_FILENAME_PATTERN.format(
                        metric_name=metric_name, extension="pickle"
                    ),
                    "wb",
                ) as f:
                    pickle.dump(optimal_fps_deltas, f, protocol=pickle.HIGHEST_PROTOCOL)

                macro_fps_deltas_dir = fps_deltas_dir.parent / "macro"

                if not macro_fps_deltas_dir.exists():
                    macro_fps_deltas_dir.mkdir(parents=True, exist_ok=True)

                with open(
                    macro_fps_deltas_dir
                    / METRIC_FILENAME_PATTERN.format(
                        metric_name=metric_name, extension="pickle"
                    ),
                    "wb",
                ) as f:
                    pickle.dump(
                        optimal_fps_deltas_avg, f, protocol=pickle.HIGHEST_PROTOCOL
                    )

        elif thresholds_to_use is None and save_files:
            interactive_heatmaps_dir = (
                METRICS_FPS_PLOTS_DIR
                / (
                    SUBMITTED_METRICS_DIRNAME
                    if not new_metrics
                    else NEW_METRICS_DIRNAME
                )
                / testset_name
                / lp
                / GOLD_SCORE_THRESHOLD_DIRNAME_PATTERN.format(
                    gold_score_threshold=gold_score_threshold
                )
                / AVERAGE_BY_DIRNAME_PATTERN.format(average_by=average_by)
                / "percentiles"
            )

            if not interactive_heatmaps_dir.exists():
                interactive_heatmaps_dir.mkdir(parents=True, exist_ok=True)

            plot_interactive_heatmap(
                prepare_heatmap_data(threshold2stats),
                thresholds,
                list(threshold2stats[next(iter(threshold2stats))].keys()),
                interactive_heatmaps_dir
                / METRIC_FILENAME_PATTERN.format(
                    metric_name=metric_name, extension="html"
                ),
            )

    return metric_name, optimization_result, optimization_result_for_qe_ranking
