from argparse import ArgumentParser
from typing import Dict, Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from mt_metrics_thresholds.definitions import (
    REGRESSION_ANALYSIS_DIR,
    SUBMITTED_METRICS_DIRNAME,
    NEW_METRICS_DIRNAME,
)
from mt_metrics_thresholds.utils.wmt import (
    get_bio_metric_name2scores,
    get_wmt_testset,
    get_wmt_metric_name2scores,
    get_grouped_metrics_scores,
)


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to perform regression analysis of MT metrics."
    )
    parser.add_argument(
        "--testset-name",
        type=str,
        choices=["wmt22", "wmt23", "bio"],
        default="wmt23",
        help="Name of the WMT test set to use. Allowed values: 'wmt22', 'wmt23', 'bio'. Default: 'wmt23'.",
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
        help="Which human reference to consider. If --testset-name or --lp is set to 'all', this param is ignored. "
        "Default: 'refA'.",
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
        help=" Whether the checkpoints of some metrics are the news ones (wrt wmt23).",
    )
    return parser


def analyze_and_plot_metrics(
    metric_name2seg_scores: Dict[str, Tuple[List[float], List[float], bool]],
    gold_name: str,
    testset_name: str,
    lp: str,
    new_metrics: bool,
    degree: int = 2,
) -> None:
    """
    Perform regression analysis of MT metrics and plot the results.

    :param metric_name2seg_scores: Dictionary from metric name to its segment-level scores.
    :param gold_name: Name of the gold scores to use in the analysis.
    :param testset_name: Name of the test set to use in the analysis.
    :param lp: Language pair to consider in the test set passed in input.
    :param new_metrics: Whether the checkpoints of some metrics are the news ones (wrt wmt23).
    :param degree: Degree of the polynomial to use in the regression analysis. Default: 3.
    """
    lp_dir = (
        REGRESSION_ANALYSIS_DIR
        / (SUBMITTED_METRICS_DIRNAME if not new_metrics else NEW_METRICS_DIRNAME)
        / testset_name
        / lp
    )
    lp_dir.mkdir(parents=True, exist_ok=True)
    with open(lp_dir / "rmse_results.txt", "w") as file:
        for metric_name, (
            metric_seg_scores,
            gold_seg_scores,
            is_qe,
        ) in tqdm(
            metric_name2seg_scores.items(), desc="Analyzing MT Metrics", unit="metric"
        ):
            if metric_name == gold_name:
                continue

            # Convert to numpy array and sort by metric scores
            metric_seg_scores = np.array(metric_seg_scores)
            gold_seg_scores = np.array(gold_seg_scores)
            sorted_indices = np.argsort(metric_seg_scores)
            metric_seg_scores = metric_seg_scores[sorted_indices]
            gold_seg_scores = gold_seg_scores[sorted_indices]

            # Linear Regression
            linear_model = LinearRegression().fit(
                metric_seg_scores.reshape(-1, 1), gold_seg_scores
            )
            xs = np.linspace(
                min(metric_seg_scores), max(metric_seg_scores), 300
            ).reshape(-1, 1)
            ys_linear = linear_model.predict(xs)

            # Polynomial Regression
            poly_features = PolynomialFeatures(degree=degree)
            metric_scores_poly = poly_features.fit_transform(
                metric_seg_scores.reshape(-1, 1)
            )
            poly_model = LinearRegression().fit(metric_scores_poly, gold_seg_scores)
            ys_poly = poly_model.predict(poly_features.transform(xs))

            # Spline Regression
            # Increase the smoothing factor 's' or set it relative to the data scale
            spline_model = UnivariateSpline(
                metric_seg_scores,
                gold_seg_scores,
                s=len(metric_seg_scores) * np.var(gold_seg_scores),
                k=3,
            )
            ys_spline = spline_model(xs[:, 0])

            # Plotting results with adjusted limits
            plt.figure(figsize=(10, 6))
            plt.scatter(
                metric_seg_scores, gold_seg_scores, color="blue", label="Original Data"
            )
            plt.plot(xs, ys_linear, "m-", label="Linear Fit")
            plt.plot(xs, ys_poly, "g--", label=f"Polynomial Fit (Degree {degree})")
            plt.plot(xs[:, 0], ys_spline, "r-", label="Spline Fit")
            plt.title(f"Regression Analysis of {metric_name}")
            plt.xlabel("MT Metric Scores")
            plt.ylabel(f"Gold {gold_name.upper()} Scores")
            plt.legend()
            plt.grid(True)
            plt.savefig(lp_dir / f"{metric_name}_regression_analysis.png")
            plt.close()

            # Print RMSE for models
            y_pred_linear = linear_model.predict(metric_seg_scores.reshape(-1, 1))
            y_pred_poly = poly_model.predict(metric_scores_poly)
            y_pred_spline = np.nan_to_num(
                spline_model(metric_seg_scores)
            )  # Replace NaNs with zero
            rmse_linear = np.sqrt(mean_squared_error(gold_seg_scores, y_pred_linear))
            rmse_poly = np.sqrt(mean_squared_error(gold_seg_scores, y_pred_poly))
            rmse_spline = np.sqrt(mean_squared_error(gold_seg_scores, y_pred_spline))

            # Write RMSE results to the file
            file.write(f"[{metric_name}] RMSE for Linear Model: {rmse_linear:.2f}\n")
            file.write(f"[{metric_name}] RMSE for Polynomial Model: {rmse_poly:.2f}\n")
            file.write(f"[{metric_name}] RMSE for Spline Model: {rmse_spline:.2f}\n")
            file.write("\n")


if __name__ == "__main__":
    parser = read_arguments()
    args = parser.parse_args()

    domains_per_seg, human_sys_names, outlier_sys_names = None, None, None
    if args.testset_name == "bio":
        metric_name2scores = get_bio_metric_name2scores(args.lp)
    else:
        testset = get_wmt_testset(args.testset_name, args.lp, True)
        metric_name2scores = get_wmt_metric_name2scores(
            testset, args.ref_to_use, args.new_metrics
        )
        domains_per_seg, human_sys_names, outlier_sys_names = (
            testset.DomainsPerSeg(),
            testset.human_sys_names,
            testset.outlier_sys_names,
        )

    metric_name2seg_scores = get_grouped_metrics_scores(
        metric_name2scores,
        args.ref_to_use,
        args.include_human,
        args.include_outliers,
        args.include_ref_to_use,
        args.gold_name,
        "none",
        domains_per_seg,
        False,
        human_sys_names,
        outlier_sys_names,
        args.testset_name == "bio",
    )

    analyze_and_plot_metrics(
        metric_name2seg_scores,
        args.gold_name,
        args.testset_name,
        args.lp,
        args.new_metrics,
    )
