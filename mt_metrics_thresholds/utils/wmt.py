import csv
import json
import logging
import pickle
import random

from typing import Dict, Tuple, List, Optional, Set, Literal, Union

from mt_metrics_thresholds.definitions import (
    METRICS_OUTPUTS_DIR,
    SEG_SCORES_FILENAME,
    SYS_SCORES_FILENAME,
    METRICS_INFO_DIR,
    METRICS_INFO_FILENAME,
    NEW_METRICS_INFO_FILENAME,
    SUBMITTED_METRICS_DIRNAME,
    NEW_METRICS_DIRNAME,
    MBR_PREDICTIONS_DIR,
)

from mt_metrics_eval import data


logger = logging.getLogger(__name__)


official_wmt_settings = {
    "wmt23": {
        "zh-en": {
            "include_human": False,
            "include_outliers": False,
        },
        "en-de": {
            "include_human": False,
            "include_outliers": False,
        },
        "he-en": {
            "include_human": True,
            "include_outliers": False,
        },
    },
    "wmt22": {
        "zh-en": {
            "include_human": False,
            "include_outliers": False,
        },
        "en-de": {
            "include_human": False,
            "include_outliers": False,
        },
        "en-ru": {
            "include_human": False,
            "include_outliers": False,
        },
    },
}


wmt_best_refs = {
    "wmt22": {
        "zh-en": "refA",
        "en-de": "refA",
        "en-ru": "refA",
    },
    "wmt23": {
        "zh-en": "refA",
        "en-de": "refA",
        "he-en": "refB",
    },
}


lp2testset_names = {
    "zh-en": ["wmt22", "wmt23"],
    "en-de": ["wmt22", "wmt23"],
    "en-ru": ["wmt22"],
    "he-en": ["wmt23"],
}


testset_name2lps = {
    "wmt22": ["zh-en", "en-de", "en-ru"],
    "wmt23": ["zh-en", "en-de", "he-en"],
}


def get_wmt_testset(
    testset_name: str, lp: str, read_stored_metric_scores: bool = False
) -> data.EvalSet:
    """
    Get the WMT test set defined by the input parameters.

    :param testset_name: Name of the WMT test set to use.
    :param lp: Language pair to consider.
    :param read_stored_metric_scores: Read stored scores for automatic metrics for this dataset.

    :return: WMT test set.
    """
    testset = data.EvalSet(testset_name, lp, read_stored_metric_scores)

    nsegs = len(testset.src)
    nsys = len(testset.sys_names)
    nmetrics = len(testset.metric_basenames)
    gold_seg = testset.StdHumanScoreName("seg")
    nrefs = len(testset.ref_names)
    std_ref = testset.std_ref

    logger.debug("\n")
    logger.debug(f"lp = {lp}.")
    logger.debug(f"# segments = {nsegs}.")
    logger.debug(f"# WMT systems = {nsys}.")
    logger.debug(f"# WMT metrics = {nmetrics}.")
    logger.debug(f"Std annotation type = {gold_seg}.")
    logger.debug(f"# refs = {nrefs}.")
    logger.debug(f"std ref = {std_ref}.")
    logger.debug("\n")

    return testset


def get_wmt_metric_name2scores(
    testset: data.EvalSet,
    ref_to_use: str,
    new_metrics: bool,
    metrics_subset: Optional[Set[str]] = None,
    gold_name: Optional[str] = None,
) -> Dict[
    str, Tuple[Dict[str, List[float]], Dict[str, List[float]], bool, Optional[str]]
]:
    """
    Read the file containing the required info for each metric and return a dictionary (for WMT test sets).

    :param testset: WMT test set.
    :param ref_to_use: Which reference to use for reference-based metrics.
    :param new_metrics: Whether the checkpoints of some metrics are the news ones (wrt wmt23).
    :param metrics_subset: Subset of metrics to consider. Default: None.
    :param gold_name: Which human ratings to use as gold scores. Default: None.

    :return: Dictionary from metric name to its scores, a boolean indicating whether the metric is a QE one, and domain.
    """
    metric_name2scores = dict()
    metrics_info_filepath = (
        METRICS_INFO_DIR
        / testset.name
        / testset.lp
        / (METRICS_INFO_FILENAME if not new_metrics else NEW_METRICS_INFO_FILENAME)
    )
    with open(metrics_info_filepath, newline="") as metrics_file:
        reader = csv.reader(metrics_file, delimiter="\t")
        for row in reader:
            # Assuming a .tsv file with four columns: Metric Name, Is Ref-less, Output Scores Dir, and Domain.
            if len(row) != 4:
                raise ValueError(
                    f"Error during parsing the file {metrics_info_filepath}, the line {row} should"
                    " contain four tab-separated elements: 'metric_name', 'is_qe', 'output_scores_dir', and 'domain'."
                )
            metric_name, is_qe, output_scores_dir, domain = row

            if (
                metrics_subset is not None
                and metric_name.lower() not in metrics_subset
                and metric_name != gold_name
            ):
                continue

            is_qe = is_qe.lower() == "yes"
            output_scores_dir = (
                METRICS_OUTPUTS_DIR
                / (
                    SUBMITTED_METRICS_DIRNAME
                    if not new_metrics
                    else NEW_METRICS_DIRNAME
                )
                / testset.name
                / testset.lp
                / output_scores_dir
            )
            if output_scores_dir.exists():
                with open(output_scores_dir / SEG_SCORES_FILENAME, "rb") as handle:
                    seg_scores = pickle.load(handle)
                with open(output_scores_dir / SYS_SCORES_FILENAME, "rb") as handle:
                    sys_scores = pickle.load(handle)
            else:
                testset_metric_name = (
                    f"{metric_name}-src"
                    if is_qe
                    else (
                        f"{metric_name}-{ref_to_use}"
                        if metric_name.lower() not in testset.human_score_names
                        else metric_name
                    )
                )
                if (
                    testset_metric_name not in testset.metric_names
                    and testset_metric_name not in testset.human_score_names
                ):
                    raise ValueError(
                        f"Metric {metric_name}'s outputs not passed in input and not present in {testset.name} "
                        f"test set with {testset.lp} lp!"
                    )
                seg_scores, sys_scores = testset.Scores(
                    "seg", testset_metric_name
                ), testset.Scores("sys", testset_metric_name)

            if metric_name in metric_name2scores:
                raise ValueError(
                    f"The metric with name {metric_name} is present more times in the input file!"
                )

            if domain not in testset.domains:
                domain = None

            metric_name2scores[metric_name] = (seg_scores, sys_scores, is_qe, domain)

    return metric_name2scores


def get_bio_metric_name2scores(
    lp: str,
) -> Dict[str, Tuple[Dict[str, List[float]], Dict[str, List[float]], bool, None]]:
    """
    Read the file containing the required info for each metric and return a dictionary (for Bio test set).

    :return: Dictionary from metric name to its scores, a boolean indicating whether the metric is a QE one, and domain.
    """
    metric_name2scores = dict()
    metrics_info_filepath = METRICS_INFO_DIR / "bio" / lp / NEW_METRICS_INFO_FILENAME
    with open(metrics_info_filepath, newline="") as metrics_file:
        reader = csv.reader(metrics_file, delimiter="\t")
        for row in reader:
            # Assuming a .tsv file with three columns: Metric Name, Is Ref-less, and Output Scores Dir.
            if len(row) != 3:
                raise ValueError(
                    f"Error during parsing the file {metrics_info_filepath}, the line {row} should"
                    " contain three tab-separated elements: 'metric_name', 'is_qe', and 'output_scores_dir'."
                )
            metric_name, is_qe, output_scores_dir = row

            is_qe = is_qe.lower() == "yes"
            output_scores_dir = (
                METRICS_OUTPUTS_DIR
                / NEW_METRICS_DIRNAME
                / "bio"
                / lp
                / output_scores_dir
            )
            with open(
                output_scores_dir
                / (
                    SEG_SCORES_FILENAME
                    if metric_name != "mqm"
                    else "sys2gold_seg_scores.pickle"
                ),
                "rb",
            ) as handle:
                seg_scores = pickle.load(handle)
            with open(
                output_scores_dir
                / (
                    SYS_SCORES_FILENAME
                    if metric_name != "mqm"
                    else "sys2gold_score.pickle"
                ),
                "rb",
            ) as handle:
                sys_scores = pickle.load(handle)

            if metric_name in metric_name2scores:
                raise ValueError(
                    f"The metric with name {metric_name} is present more times in the input file!"
                )

            metric_name2scores[metric_name] = (seg_scores, sys_scores, is_qe, None)

    return metric_name2scores


def get_n_segs_with_gold_annotation(testset: data.EvalSet, gold_name: str) -> int:
    """
    Get the number of segments with gold `gold_name` annotations for a given WMT test set.

    :param testset: WMT test set.
    :param gold_name: Which human ratings to consider as gold scores.

    :return: Number of segments with gold `gold_name` annotations in the input WMT test set.
    """
    return sum(
        gt is not None for gt in next(iter(testset.Scores("seg", gold_name).values()))
    )


def filter_out_mt_systems(
    sys2scores: Dict[str, List[float]],
    ref_to_use: str,
    include_human: bool,
    include_outliers: bool,
    include_ref_to_use: bool,
    human_sys_names: Optional[Set[str]] = None,
    outlier_sys_names: Optional[Set[str]] = None,
) -> Dict[str, List[float]]:
    """
    Filter out reference system, human systems, and outlier systems according to the input parameters.

    :param sys2scores: Dictionary from system name to its segment-level scores.
    :param ref_to_use: Which human reference to consider.
    :param include_human: Whether to include 'human' systems (i.e., reference translations) among systems.
    :param include_outliers: Whether to include systems considered to be outliers.
    :param include_ref_to_use: Whether to include the reference system (for QE metrics).
    :param human_sys_names: Set of human system names. Default: None.
    :param outlier_sys_names: Set of outlier system names. Default: None.

    :return: Filtered dictionary from system name to its segment-level scores.
    """
    filtered_sys2scores = sys2scores.copy()

    if not include_human:
        for sys in human_sys_names:
            filtered_sys2scores.pop(sys, None)
    if not include_outliers:
        for sys in outlier_sys_names:
            filtered_sys2scores.pop(sys, None)

    if not include_ref_to_use:
        filtered_sys2scores.pop(ref_to_use, None)

    return filtered_sys2scores


def get_grouped_metrics_scores(
    metric_name2scores: Dict[
        str, Tuple[Dict[str, List[float]], Dict[str, List[float]], bool, Optional[str]]
    ],
    ref_to_use: str,
    include_human: bool,
    include_outliers: bool,
    include_ref_to_use: bool,
    gold_name: str,
    average_by: Literal["none", "item", "sys"],
    domains_per_seg: Optional[List[str]] = None,
    return_valid_systems: bool = False,
    human_sys_names: Optional[Set[str]] = None,
    outlier_sys_names: Optional[Set[str]] = None,
    is_bio: bool = False,
    sample_size: Optional[int] = None,
) -> Union[
    Tuple[
        Dict[str, Tuple[List[List[float]], List[List[float]], bool]],
        List[str],
        List[str],
    ],
    Union[
        Dict[str, Tuple[List[float], List[float], bool]],
        Dict[str, Tuple[List[List[float]], List[List[float]], bool]],
    ],
]:
    """
    Get the grouped segment-level scores for each metric.

    :param metric_name2scores: Dictionary from metric name to its scores and is_qe flag.
    :param ref_to_use: Which human reference to consider.
    :param include_human: Whether to include 'human' systems (i.e., reference translations) among systems.
    :param include_outliers: Whether to include systems considered to be outliers.
    :param include_ref_to_use: Whether to include the reference system (for QE metrics).
    :param gold_name: Which human ratings to use as gold scores.
    :param average_by: How to group the segment-level scores for each metric. Allowed values: "none", "item", "sys".
    :param domains_per_seg: List containing the domain string for each segment. Default: None.
    :param return_valid_systems: Whether to return the valid systems (used only in item-grouping). Default: False.
    :param human_sys_names: Set of human system names. Default: None.
    :param outlier_sys_names: Set of outlier system names. Default: None.
    :param is_bio: Whether the test set is Bio. Default: False.
    :param sample_size: If not None, it defines the number of items to sample. Default: None.

    :return: Dictionary from metric name to its grouped segment-level scores and, optionally, also the valid systems.
    """
    if average_by != "none" and average_by != "item" and average_by != "sys":
        raise ValueError(
            f"Invalid value for 'average_by' parameter: {average_by}. Allowed values: 'none', 'item', 'sys'."
        )
    if return_valid_systems and average_by != "item":
        raise ValueError("return_valid_systems is used only in item-grouping!")
    if sample_size is not None and average_by != "sys":
        raise ValueError("Sampling segments is supported only in sys-grouping!")

    metric_name2grouped_seg_scores = dict()

    sys2gold_seg_scores = (
        filter_out_mt_systems(
            metric_name2scores[gold_name][0],
            ref_to_use,
            include_human,
            include_outliers,
            include_ref_to_use,
            human_sys_names,
            outlier_sys_names,
        )
        if not is_bio
        else metric_name2scores[gold_name][0]
    )

    # valid_systems_qe and valid_systems are used only for item and sys grouping. Sorted to ensure consistent order.
    valid_systems_qe = sorted(
        [
            sys
            for sys, gold_seg_scores in sys2gold_seg_scores.items()
            if any(score is not None for score in gold_seg_scores)
        ]
    )
    valid_systems = (
        [sys for sys in valid_systems_qe if sys != ref_to_use]
        if not is_bio
        else valid_systems_qe.copy()
    )

    n_segs = len(sys2gold_seg_scores[valid_systems_qe[0]])
    for sys in valid_systems_qe[1:]:
        if len(sys2gold_seg_scores[sys]) != n_segs:
            raise ValueError(
                f"For {valid_systems_qe[0]} sys there are {n_segs} segment-level gold {gold_name} scores, but for {sys}"
                f" sys there are {len(sys2gold_seg_scores[sys])} segment-level gold {gold_name} scores!"
            )

    sample_indexes = set()
    if sample_size is None:
        print("\n")
        print(f"# valid MT systems for ref-based metrics = {len(valid_systems)}.")
        print(f"# valid MT systems for QE metrics = {len(valid_systems_qe)}.")
        print(f"# segments = {n_segs}.")
        print("\n")
    else:
        valid_indexes = [
            seg_idx
            for seg_idx, gt in enumerate(sys2gold_seg_scores[valid_systems_qe[0]])
            if gt is not None
        ]
        if sample_size > len(valid_indexes):
            raise ValueError(
                f"Sample size {sample_size} is greater than the number of segments with human {gold_name} annotation "
                f"({len(valid_indexes)})!"
            )
        sample_indexes = set(random.sample(valid_indexes, sample_size))

    # Process each metric
    for metric_name, (
        sys2metric_seg_scores,
        sys2metric_score,
        is_qe,
        domain,
    ) in metric_name2scores.items():
        n_segs_for_metric, valid_systems_for_metric, sys2gold_seg_scores_for_metric = (
            n_segs,
            valid_systems if not is_qe else valid_systems_qe,
            sys2gold_seg_scores,
        )
        n_sys_for_metric = len(valid_systems_for_metric)

        if domain is not None:
            if len(domains_per_seg) != n_segs:
                raise ValueError(
                    f"Domains per segment are not provided for all segments for metric {metric_name}!"
                )
            if sample_size is not None:
                raise ValueError(
                    "Domain filtering is not supported when sampling segments!"
                )

            sys2gold_seg_scores_for_metric = dict()
            for sys, gold_seg_scores in sys2gold_seg_scores.items():
                sys2gold_seg_scores_for_metric[sys] = [
                    score
                    for score, d in zip(gold_seg_scores, domains_per_seg)
                    if d == domain
                ]

            n_segs_for_metric = len(
                sys2gold_seg_scores_for_metric[valid_systems_for_metric[0]]
            )

        if average_by == "none":
            grouped_metric_seg_scores, grouped_gold_seg_score = [], []
        else:
            n_groups = n_segs_for_metric if average_by == "item" else n_sys_for_metric
            grouped_metric_seg_scores, grouped_gold_seg_score = [
                [] for _ in range(n_groups)
            ], [[] for _ in range(n_groups)]

        for sys, metric_seg_scores in sys2metric_seg_scores.items():
            if len(metric_seg_scores) != n_segs_for_metric:
                raise ValueError(
                    f"For {sys} sys, the {metric_name} metric has {len(metric_seg_scores)} segment-level scores, "
                    f"while there are {n_segs_for_metric} segments with gold {gold_name} scores!"
                )

        if average_by == "none":
            # most prob this is old, and we've to take into account `valid_systems_for_metric` now
            for sys, gold_seg_scores in sys2gold_seg_scores_for_metric.items():
                if sys == ref_to_use and not is_qe and not is_bio:
                    continue

                for gt, score in zip(gold_seg_scores, sys2metric_seg_scores[sys]):
                    if gt is not None:
                        if score is None:
                            if metric_name != "da-sqm":
                                raise ValueError(
                                    f"For {metric_name} metric, a seg score for {sys} sys is None for a valid gold "
                                    f"scores!"
                                )
                            else:
                                continue

                        grouped_metric_seg_scores.append(score)
                        grouped_gold_seg_score.append(gt)

        elif average_by == "item":
            # Filter and order scores based on valid and sorted systems
            for sys in valid_systems_for_metric:
                gold_seg_scores, metric_seg_scores = (
                    sys2gold_seg_scores_for_metric[sys],
                    sys2metric_seg_scores[sys],
                )
                for seg_idx, (gt, score) in enumerate(
                    zip(gold_seg_scores, metric_seg_scores)
                ):
                    if gt is not None:
                        if score is None:
                            if metric_name != "da-sqm":
                                raise ValueError(
                                    f"For {metric_name} metric, a seg score for {sys} sys is None for a valid gold "
                                    f"scores!"
                                )
                            else:
                                continue

                        grouped_metric_seg_scores[seg_idx].append(score)
                        grouped_gold_seg_score[seg_idx].append(gt)

            print("\n")
            print(
                f"# segments with at least one non-None annotation for {metric_name} metric = "
                f"{sum(1 for g in grouped_gold_seg_score if len(g) > 0)}"
            )
            print(
                f"# segments with at least two non-None annotations for {metric_name} metric = "
                f"{sum(1 for g in grouped_gold_seg_score if len(g) > 1)}"
            )
            print("\n")

        else:
            for sys_idx, sys in enumerate(valid_systems_for_metric):
                gold_seg_scores, metric_seg_scores = (
                    sys2gold_seg_scores_for_metric[sys],
                    sys2metric_seg_scores[sys],
                )
                for seg_idx, (gt, score) in enumerate(
                    zip(gold_seg_scores, metric_seg_scores)
                ):
                    if gt is not None:
                        if score is None:
                            if metric_name != "da-sqm":
                                raise ValueError(
                                    f"For {metric_name} metric, a seg score for {sys} sys is None for a valid gold "
                                    f"scores!"
                                )
                            else:
                                continue

                        if sample_size is not None and seg_idx not in sample_indexes:
                            continue

                        grouped_metric_seg_scores[sys_idx].append(score)
                        grouped_gold_seg_score[sys_idx].append(gt)

        metric_name2grouped_seg_scores[metric_name] = (
            grouped_metric_seg_scores,
            grouped_gold_seg_score,
            is_qe,
        )

    return (
        (metric_name2grouped_seg_scores, valid_systems, valid_systems_qe)
        if return_valid_systems
        else metric_name2grouped_seg_scores
    )


def get_mbr_data(
    lp: str,
    ref_to_use: str,
    include_human: bool,
    include_outliers: bool,
    gold_name: str,
    new_metrics: bool,
    reverse_mbr: bool,
) -> Tuple[Dict[str, Dict[int, List[str]]], Dict[int, Dict[str, float]]]:
    """
    Read the metrics MBR data from the corresponding WMT23 lp .jsonl files and return a dictionary, together with GT.

    :param lp: Language pair to consider for the WMT23 test set.
    :param ref_to_use: Which human reference to consider.
    :param include_human: Whether to include 'human' systems (i.e., reference translations) among systems.
    :param include_outliers: Whether to include systems considered to be outliers.
    :param gold_name: Which human ratings to use as gold scores.
    :param new_metrics: Whether the checkpoints of some metrics are the news ones (wrt WMT23).
    :param reverse_mbr: Whether to apply reverse MBR.

    :return: A dictionary from metric names to their MBR predictions in the input WMT23 test set, together with GT.
    """
    testset = get_wmt_testset("wmt23", lp, True)

    sys2gold_seg_scores = filter_out_mt_systems(
        testset.Scores("seg", gold_name),
        ref_to_use,
        include_human,
        include_outliers,
        False,
        testset.human_sys_names,
        testset.outlier_sys_names,
    )
    seg_idx2gold_sys_scores = dict()
    for sys, gold_seg_scores in sys2gold_seg_scores.items():
        for seg_idx, gold_score in enumerate(gold_seg_scores):
            if gold_score is None:
                if seg_idx in seg_idx2gold_sys_scores:
                    raise ValueError(
                        f"The segment with index {seg_idx} in WMT23 for {lp} lp has missing gold {gold_name} scores "
                        f"only for some MT systems!"
                    )
                continue
            elif seg_idx not in seg_idx2gold_sys_scores:
                seg_idx2gold_sys_scores[seg_idx] = dict()
            seg_idx2gold_sys_scores[seg_idx][sys] = gold_score

    mbr_predictions_lp_dir = (
        MBR_PREDICTIONS_DIR
        / (SUBMITTED_METRICS_DIRNAME if not new_metrics else NEW_METRICS_DIRNAME)
        / "wmt23"
        / lp
    )
    metric_name2mbr_predictions = dict()

    def process_segment(_prev_seg_idx: int, _metric_name: str) -> None:
        """
        Process the segment with `_prev_seg_idx` index for the `_metric_name` MT metric, adding its MBR predictions.

        :param _prev_seg_idx: Index of the segment to process.
        :param _metric_name: Name of the MT metric for which to add MBR predictions.
        """
        max_mbr_score, max_sys_names = float("-inf"), []
        prev_n_scores = None

        for sys_name, predictions in metric_name2mbr_predictions[_metric_name][
            _prev_seg_idx
        ].items():
            mbr_score = sum(predictions) / len(predictions)

            if prev_n_scores is not None and len(predictions) != prev_n_scores:
                raise ValueError(
                    f"For the segment with index {_prev_seg_idx} in WMT23 for {lp} lp, the number of "
                    f"{_metric_name} metric scores is not the same across the several MT systems!"
                )
            prev_n_scores = len(predictions)

            if mbr_score == max_mbr_score:
                max_sys_names.append(sys_name)
            elif mbr_score > max_mbr_score:
                max_mbr_score = mbr_score
                max_sys_names = [sys_name]

        assert len(max_sys_names) > 0

        if _prev_seg_idx not in seg_idx2gold_sys_scores:
            raise ValueError(
                f"The segment with index {_prev_seg_idx} in WMT23 for {lp} lp does not have gold "
                f"{gold_name} scores, but it has MBR predictions for the {_metric_name} metric!"
            )
        metric_name2mbr_predictions[_metric_name][_prev_seg_idx] = max_sys_names

    for metric_mbr_prediction_file in mbr_predictions_lp_dir.glob("*.jsonl"):
        metric_name = metric_mbr_prediction_file.stem
        metric_name2mbr_predictions[metric_name] = dict()
        with open(metric_mbr_prediction_file, "r", encoding="utf-8") as file:
            prev_seg_idx = None
            for line in file:
                sample = json.loads(line)

                if prev_seg_idx is not None and sample["seg_idx"] != prev_seg_idx:
                    process_segment(prev_seg_idx, metric_name)
                prev_seg_idx = sample["seg_idx"]

                if (
                    sample["hyp_sys_name"] not in sys2gold_seg_scores
                    or sample["ref_sys_name"] not in sys2gold_seg_scores
                ):
                    continue  # There are some MT systems that are not evaluated with gold scores.

                if sample["seg_idx"] not in metric_name2mbr_predictions[metric_name]:
                    metric_name2mbr_predictions[metric_name][sample["seg_idx"]] = dict()
                sys_to_score = "hyp_sys_name" if not reverse_mbr else "ref_sys_name"
                if (
                    sample[sys_to_score]
                    not in metric_name2mbr_predictions[metric_name][sample["seg_idx"]]
                ):
                    metric_name2mbr_predictions[metric_name][sample["seg_idx"]][
                        sample[sys_to_score]
                    ] = []
                metric_name2mbr_predictions[metric_name][sample["seg_idx"]][
                    sample[sys_to_score]
                ].append(sample["prediction"])

            process_segment(prev_seg_idx, metric_name)

    gold_seg_ids = set(seg_idx2gold_sys_scores.keys())
    for metric_name, mbr_predictions in metric_name2mbr_predictions.items():
        metric_pred_seg_ids = set(mbr_predictions)
        if gold_seg_ids != metric_pred_seg_ids:
            raise ValueError(
                f"{metric_name} metric has distinct segment indexes compared to gold {gold_name} annotations for its "
                f"MBR predictions in WMT23 for {lp} lp!"
            )

    return metric_name2mbr_predictions, seg_idx2gold_sys_scores
