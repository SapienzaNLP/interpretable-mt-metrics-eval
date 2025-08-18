import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
import argparse
from scripts_utils import read_metric2latex

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.sans-serif": ["Computer Modern Sans"],
    "font.monospace": ["Computer Modern Typewriter"],
    "axes.unicode_minus": False  # Avoid using Unicode minus
})

mainpaper_metrics = [
    "XCOMET-XL",
    "MetricX-23-XL",
    "MaTESe",
    "COMET",
    "BLEURT-20",
    "MetricX-23-QE-XL",
    "COMET-QE-MQM",
    "COMET-QE",
    "CometKiwi",
    "CometKiwi-XL",
    "GEMBA-MQM",
    "MaTESe-QE",
    "da-sqm"
]

metric2latex = read_metric2latex()

def reorder_and_rename(data, apx):
    new_data = {}
    for metric in metric2latex:
        if not apx:
            if metric in mainpaper_metrics and metric in data:
                new_data[metric2latex[metric]] = data[metric]
        else:                
            if metric in data:
                new_data[metric2latex[metric]] = data[metric]
    return new_data

def plot_fps_deltas(lp, dataset, apx):

    fps_data_1 = {}
    fps_data_2 = {}

    dirpath_1 = f"metrics_data/fps_deltas/new_metrics/{dataset}/{lp}/gold_score_threshold_-1.0/sys_grouping/macro/"
    dirpath_2 = f"metrics_data/fps_deltas/new_metrics/{dataset}/{lp}/gold_score_threshold_-4.0/sys_grouping/macro/"

    def load_data(dirpath, fps_data, metric2latex):
        for filename in os.listdir(dirpath):
            filepath = os.path.join(dirpath, filename)
            metric_name = filename.split(".")[0]
            if filename.endswith(".pickle"):
                with open(filepath, 'rb') as fin:
                    data = pickle.load(fin)
                    cr_name = metric2latex.get(metric_name, None)
                    if not cr_name:
                        continue
                    fps_data[metric_name] = data

    load_data(dirpath_1, fps_data_1, metric2latex)
    load_data(dirpath_2, fps_data_2, metric2latex)

    fps_data_1 = reorder_and_rename(fps_data_1, apx)
    fps_data_2 = reorder_and_rename(fps_data_2, apx)

    # Creating DataFrames from the vectors
    df1 = pd.DataFrame(fps_data_1)
    df1['Source'] = "\\textsc{Perfect}" 

    df2 = pd.DataFrame(fps_data_2)
    df2['Source'] = "\\textsc{Good}"

    
    print(f"The minimum avg value is {min(df1.iloc[:, :-2].mean().min(), df2.iloc[:, :-2].mean().min())}")
    print(f"The absolute maximum is {max(df1.iloc[:, :-2].mean().max(), df2.iloc[:, :-2].mean().max())}")

    custom_palette = {
        "\\textsc{Perfect}" : (0.75, 0.75, 1.0), #'#B3D1FF', 
        "\\textsc{Good}":  (0.75, 1.0, 0.75), #'#B3FFB3' 
    }

    # Combining the DataFrames
    df_combined = pd.concat([df1, df2])

    # Melting the DataFrame for seaborn compatibility
    df_melted = df_combined.melt(id_vars=['Source'], var_name='Metric', value_name='Score')


    # Plotting the violin plot
    if args.apx:
        plt.figure(figsize=(5, 12))
    else:
        plt.figure(figsize=(4, 5))
    sns.violinplot(data=df_melted, x='Score', y='Metric', hue='Source', split=True, inner='quart', palette=custom_palette)
    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))

    plt.xlabel("False Positive MQM Score $\\Delta$")
    plt.ylabel("")

    plt.tight_layout()

    apx_str = "-apx" if apx else ""
    plt.savefig(f"metrics_data/tex_artifacts/fps-deltas-binary-{dataset}-{lp.split('-')[0]+lp.split('-')[1]}{apx_str}.pdf", format='pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language-pair", "-lp", type=str, default="zh-en", help="Language pair")
    parser.add_argument("--dataset", "-d", type=str, default="wmt23", help="Dataset")
    parser.add_argument("--apx", "-a", action="store_true", help="Appendix")
    args = parser.parse_args()

    plot_fps_deltas(args.language_pair, args.dataset, args.apx)
