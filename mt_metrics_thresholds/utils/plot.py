from pathlib import Path
from typing import List, Union, Literal, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

import plotly.graph_objects as go
import plotly.offline as pyo


def save_data_distribution_plot(
    data: List[float],
    bins: Union[int, str],
    plot_title: str,
    x_axis_name: str,
    y_axis_name: str,
    file_path: Path,
    add_statistics_to_title: bool = False,
) -> None:
    """
    Compute and save the plot of the input data distribution.

    :param data: Input float data to plot.
    :param bins: Number of bins to use in the histograms (or a string rule).
    :param plot_title: Title of the plot.
    :param x_axis_name: Name of the x-axis.
    :param y_axis_name: Name of the y-axis.
    :param file_path: Path to save the plot.
    :param add_statistics_to_title: Whether to add statistics (mean, median, mode, std dev, etc.) to the title.
    """
    sns.set_theme(style="whitegrid")  # Setting the seaborn style for all plots

    plt.figure(figsize=(10, 6))  # Set the figure size for better readability
    sns.histplot(data, kde=True, stat="density", linewidth=0, bins=bins)

    if add_statistics_to_title:
        # Calculating necessary statistics
        mean_val = np.mean(data)
        median_val = np.median(data)
        mode_val = stats.mode(data).mode
        std_dev = np.std(data)
        skewness = stats.skew(np.array(data))
        kurtosis_val = stats.kurtosis(data)

        plot_title = (
            f"{plot_title}\n"
            f"Mean: {mean_val:.2f}, Median: {median_val:.2f}, Mode: {mode_val:.2f}, "
            f"Std Dev: {std_dev:.2f}, Skewness: {skewness:.2f}, Kurtosis: {kurtosis_val:.2f}"
        )
    plt.title(plot_title)

    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)

    plt.savefig(file_path)  # Save the figure
    plt.close()  # Close the figure to free up memory


def plot_precision_recall_curve(
    metric_name: str,
    average_by: Literal["none", "item", "sys"],
    file_path: Path,
    metric_scores: Optional[List[float]] = None,
    gold_scores: Optional[List[float]] = None,
    precisions: Optional[List[float]] = None,
    recalls: Optional[List[float]] = None,
) -> Tuple[float, float, float]:
    """
    Plot the Precision-Recall curve, and return AP and AUC (w/ and w/o artificial endpoint).

    :param metric_name: Name of the MT metric.
    :param average_by: What to average over when computing final scores. Allowed values: 'none', 'item', 'sys'.
    :param file_path: Path to save the plot.
    :param metric_scores: List of scores from the MT metric.
    :param gold_scores: List of gold scores.
    :param precisions: Precisions (increasing thresholds). If passed, `metric_scores` and `gold_scores` are not used.
    :param recalls: Recalls (increasing thresholds). If passed, `metric_scores` and `gold_scores` are not used.

    :return: AP and AUC of the Precision-Recall curve (w/ and w/o artificial endpoint).
    """
    if average_by != "none" and average_by != "item" and average_by != "sys":
        raise ValueError(
            f"Invalid average_by parameter: {average_by}. Allowed values: 'none', 'item'."
        )

    # Calculate Precision-Recall curve, AP, and AUC
    if average_by == "item" or average_by == "sys" or precisions is not None:
        if len(precisions) != len(recalls):
            raise ValueError(
                f"Precision and Recall lists must have the same length! Precision: {len(precisions)}, Recall: "
                f"{len(recalls)}."
            )

        if precisions[-1] != 1 or recalls[-1] != 0:
            precisions.append(1)
            recalls.append(0)

        ap = -np.sum(np.diff(recalls) * np.array(precisions)[:-1])

    else:
        precisions, recalls, thresholds = precision_recall_curve(
            gold_scores, metric_scores
        )
        assert len(precisions) == len(recalls) == len(thresholds) + 1
        ap = average_precision_score(gold_scores, metric_scores)

    auc_pr = auc(recalls, precisions)
    # Exclude the last point (artificial endpoint) for AUC computation
    auc_pr_no_artificial_endpoint = auc(recalls[:-1], precisions[:-1])

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=recalls, y=precisions, drawstyle="steps-post")
    plt.title(
        f"Precision-Recall Curve for {metric_name} (AP: {ap:.2f}, AUC: {auc_pr:.2f}, AUC (no artificial endpoint): "
        f"{auc_pr_no_artificial_endpoint:.2f})"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    # Save plot
    plt.savefig(file_path)
    plt.close()

    return ap, auc_pr, auc_pr_no_artificial_endpoint


def normalize_data_for_color(data_array: np.ndarray) -> np.ndarray:
    """
    Normalize data for color scaling in heatmap.

    :param data_array: Data to normalize with min-max scaling.

    :return: Normalized data.
    """
    normalized_data = np.zeros(data_array.shape)
    for i, row in enumerate(data_array):
        min_val = np.min(row)
        max_val = np.max(row)
        # Avoid division by zero if min and max are the same
        if min_val == max_val:
            # If all values in a row are the same, set normalized value to a default or middle value,
            # for example 0.5 to indicate neutral color
            normalized_data[i] = 0.5
        else:
            normalized_data[i] = (row - min_val) / (max_val - min_val)

    return normalized_data


def prepare_heatmap_data(
    threshold2stats: Dict[float, Dict[str, float]]
) -> List[List[float]]:
    """
    Prepare data for heatmap visualization based on the input stats.

    :param threshold2stats: Dictionary containing the stats for each threshold.

    :return: 2D list of data ready for heatmap plotting.
    """
    # Initialize an empty list to hold the data for each stat
    data = []

    # Iterate over each stat to fill the data array
    for stat_name in list(threshold2stats[next(iter(threshold2stats))].keys()):
        stat_data = []
        for threshold in threshold2stats:
            stat_value = threshold2stats[threshold][stat_name]
            stat_data.append(stat_value)
        data.append(stat_data)

    return data


def plot_interactive_heatmap(
    data: List[List[float]],
    threshold_values: List[float],
    stat_names: List[str],
    file_path: Union[Path, str],
    figure_width: int = 2000,
    figure_height: int = 600,
) -> None:
    """
    Generate and save an interactive heatmap for stats across thresholds to an HTML file.

    :param data: 2D list of data ready for heatmap plotting.
    :param threshold_values: List of threshold values to use on the x-axis.
    :param stat_names: List of stat names to include in the heatmap.
    :param file_path: Path to save the HTML file containing the heatmap.
    :param figure_width: Width of the heatmap figure. Default: 2000.
    :param figure_height: Height of the heatmap figure. Default: 600.
    """
    # Convert data to numpy array for processing
    data_array = np.array(data)
    normalized_data = normalize_data_for_color(data_array)

    # Setting up the x-axis as categorical with evenly spaced labels
    tick_positions = list(range(len(threshold_values)))  # Positions for each tick

    fig = go.Figure(
        data=go.Heatmap(
            z=normalized_data,
            x=tick_positions,  # Use positions for x-coordinates
            y=stat_names,
            colorscale="Viridis",
            hoverinfo="skip",
            text=np.around(data_array, decimals=2).astype(str),
            texttemplate="%{text}",
        )
    )

    fig.update_layout(
        title="Heatmap of Stats Across Threshold Values",
        xaxis=dict(
            title="Threshold Values",
            tickmode="array",
            tickvals=tick_positions,  # Positions of the ticks
            ticktext=[
                f"{round(val, 3)}" for val in threshold_values
            ],  # Texts are actual threshold values
        ),
        yaxis_title="Stats",
        width=figure_width,
        height=figure_height,
    )

    pyo.plot(fig, filename=str(file_path), auto_open=False)
