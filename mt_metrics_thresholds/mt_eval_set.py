import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Dict, List, Literal

from mt_metrics_eval import data

from mt_metrics_thresholds.definitions import (
    SEG_SCORES_FILENAME,
    SYS_SCORES_FILENAME,
    METRICS_OUTPUTS_DIR,
    HUMAN_SYS_NAMES_FILE,
    OUTLIER_SYS_NAMES_FILE,
    HUMAN_SCORE_NAMES_FILE,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MTMetricOutputs:
    sys2seg_scores: Dict[str, List[float]]
    sys2score: Dict[str, List[float]]


class MTEvalSet:
    """
    A class to represent an MT evaluation set.

    :param path: Path to the MT evaluation set directory.
    :param name: Name of the MT evaluation set. Required if `wmt_eval_set` is left to None. Default: None.
    :param lp: Language pair, e.g., `'en-de'`. Required if `wmt_eval_set` is left to None. Default: None.
    :param wmt_eval_set: WMT evaluation set. If set, `name` and `lp` are ignored, using this object. Default: None.

    :raises ValueError: If neither `wmt_eval_set` nor the combination of `name` and `lp` are provided.
    """

    def __init__(
        self,
        path: Path,
        name: Optional[str] = None,
        lp: Optional[str] = None,
        wmt_eval_set: Optional[data.EvalSet] = None,
    ):
        if wmt_eval_set is None and name is None and lp is None:
            raise ValueError(
                "Either provide `wmt_eval_set` or the combination of `name` and `lp`!"
            )

        self.path = path
        self.wmt_eval_set = wmt_eval_set
        self.name = self.wmt_eval_set.name if self.wmt_eval_set else name
        self.lp = self.wmt_eval_set.lp if self.wmt_eval_set else lp

        self._metric_basenames = (
            self.wmt_eval_set.metric_basenames
            if self.wmt_eval_set is not None
            else set()
        )
        self._wmt_metric_basenames = (
            self.wmt_eval_set.metric_basenames
            if self.wmt_eval_set is not None
            else set()
        )
        self._human_score_names = (
            self.wmt_eval_set.human_score_names
            if self.wmt_eval_set is not None
            else set()
        )

        self._human_sys_names = (
            self.wmt_eval_set.human_sys_names
            if self.wmt_eval_set is not None
            else set()
        )
        self._outlier_sys_names = (
            self.wmt_eval_set.outlier_sys_names
            if self.wmt_eval_set is not None
            else set()
        )

        self._load_data_from_files()

    @property
    def metric_basenames(self) -> Set[str]:
        """Basenames of all the available metrics, e.g. BLEU, COMET."""
        return self._metric_basenames

    @property
    def wmt_metric_basenames(self) -> Set[str]:
        """Basenames of available metrics for which the outputs have not been reloaded."""
        return self._wmt_metric_basenames

    @property
    def human_score_names(self) -> Set[str]:
        """Names of different human scores available, eg `'wmt-z'`, `'mqm'`."""
        return self._human_score_names

    @property
    def human_sys_names(self) -> Set[str]:
        """Names of human systems."""
        return self._human_sys_names

    @property
    def outlier_sys_names(self) -> Set[str]:
        """Names of systems considered to be outliers."""
        return self._outlier_sys_names

    def _load_data_from_files(self) -> None:
        """
        Load the metrics' outputs from the corresponding files.

        :raises NotADirectoryError: If a file is found instead of a directory for metric output scores.
        """
        self._loaded_metric_name2outputs = dict()
        lp_dir = self.path / self.lp

        human_score_names_filepath = lp_dir / HUMAN_SCORE_NAMES_FILE
        if human_score_names_filepath.exists():
            with open(human_score_names_filepath, "r") as file:
                logger.warning(
                    f"Human score names file found for {self.name} WMT evaluation set in {self.lp} language pair. This "
                    "will override the ones in it."
                )
                self._human_score_names = set(file.read().splitlines())

        metrics_outputs_dirpath = lp_dir / METRICS_OUTPUTS_DIR
        self._metric_basenames = list(self._metric_basenames)
        for metric_outputs_dir in metrics_outputs_dirpath.iterdir():
            if not metric_outputs_dir.is_dir():
                raise NotADirectoryError(
                    "Expected a directory containing output scores for each metric, but found a file: "
                    f"{metric_outputs_dir}"
                )

            with open(metric_outputs_dir / SEG_SCORES_FILENAME, "rb") as handle:
                sys2seg_scores = pickle.load(handle)
            with open(metric_outputs_dir / SYS_SCORES_FILENAME, "rb") as handle:
                sys2score = pickle.load(handle)

            metric_name = metric_outputs_dir.name
            self._loaded_metric_name2outputs[metric_name] = MTMetricOutputs(
                sys2seg_scores, sys2score
            )

            self._metric_basenames.append(metric_name)
            self._wmt_metric_basenames.discard(metric_name)
        self._metric_basenames = set(self._metric_basenames)

        for sys_names_filepath in [
            lp_dir / HUMAN_SYS_NAMES_FILE,
            lp_dir / OUTLIER_SYS_NAMES_FILE,
        ]:
            if sys_names_filepath.exists():
                if self.wmt_eval_set is not None:
                    logger.warning(
                        f"System names file found for {self.name} WMT evaluation set in {self.lp} language pair. This "
                        "will override the ones in it."
                    )
                with open(sys_names_filepath, "r") as file:
                    sys_names = set(file.read().splitlines())
                    if sys_names_filepath.name == HUMAN_SYS_NAMES_FILE:
                        self._human_sys_names = sys_names
                    else:
                        self._outlier_sys_names = sys_names

    def Scores(
        self, level: Literal["seg", "sys"], scorer: str
    ) -> Dict[str, List[float]]:
        """
        Get stored scores assigned to text units at a given level.

        :param level: Text units to which scores apply, one of `'sys'` or `'seg'`.
        :param scorer: Method used to produce scores. If it is a non-reloaded WMT metric, it must contain also the ref.

        :return: Mapping from system names to lists of float scores.

        :raise ValueError: If an invalid level is specified or the scorer isn't available.
        """
        if level != "seg" and level != "sys":
            raise ValueError(
                f"Invalid level specified ({level})! Choose from 'sys', 'seg'."
            )
        if scorer in self._loaded_metric_name2outputs:
            return (
                self._loaded_metric_name2outputs[scorer].sys2seg_scores
                if level == "seg"
                else self._loaded_metric_name2outputs[scorer].sys2score
            )
        elif (scorer in self.wmt_eval_set.metric_names) or (
            scorer in self.wmt_eval_set.human_score_names
        ):
            return self.wmt_eval_set.Scores(level, scorer)
        else:
            raise ValueError(f"Scorer {scorer} not available in the evaluation set!")
