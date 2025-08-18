import json
import argparse
from scripts_utils import read_metric2command


metric2command = read_metric2command()

metric2highlight = {
    "XCOMET-Ensemble": True,
    "XCOMET-XL": False,
    "MetricX-23-QE-XL": False,
    "MetricX-23-QE": True,
    "MetricX-23-XL": False,
    "MetricX-23": True,
    "XCOMET-QE-Ensemble": True,
    "GEMBA-MQM": False,
    "COMET-QE-MQM": False,
    "CometKiwi": False,
    "CometKiwi-XL": False,
    "COMET-QE": False,
    "COMET": False,
    "BLEURT-20": False,
    "MaTESe": False,
    "MaTESe-QE": False,
    "mbr-metricx-qe": True,
    "da-sqm": True,
    "BERTscore": False,
    "eBLEU": False,
    "f200spBLEU": False,
    "BLEU": False,
    "chrF": False,
    "tokengram_F": False,
    "Random-sysname": True,
    "SENTINEL-SRC-MQM": False,
    "SENTINEL-REF-MQM": False,
    "SENTINEL-CAND-MQM": False,
}

metric2category = {
    "XCOMET-Ensemble": "reference based",
    "XCOMET-XL": "reference based",
    "MetricX-23-QE-XL": "reference free",
    "MetricX-23-QE": "reference free",
    "MetricX-23-XL": "reference based",
    "MetricX-23": "reference based",
    "XCOMET-QE-Ensemble": "reference free",
    "GEMBA-MQM": "reference free",
    "COMET-QE-MQM": "reference free",
    "CometKiwi": "reference free",
    "CometKiwi-XL": "reference free",
    "COMET-QE": "reference free",
    "COMET": "reference based",
    "BLEURT-20": "reference based",
    "MaTESe": "reference based",
    "MaTESe-QE": "reference free",
    "mbr-metricx-qe": "reference free",
    "BERTscore": "reference based",
    "BLEU": "lexical based",
    "chrF": "lexical based",
    "f200spBLEU": "lexical based",
    "eBLEU": "lexical based",
    "tokengram_F": "lexical based",
    "SENTINEL-SRC-MQM": "sentinel metrics",
    "SENTINEL-REF-MQM": "sentinel metrics",
    "SENTINEL-CAND-MQM": "sentinel metrics",
}

category2metrics = {
    "reference based": [
        "XCOMET-Ensemble",
        "XCOMET-XL",
        "MetricX-23",
        "MetricX-23-XL",
        "MaTESe",
        "COMET",
        "BLEURT-20",
        "BERTscore"
    ],
    "reference free": [
        "XCOMET-QE-Ensemble",
        "MetricX-23-QE",
        "MetricX-23-QE-XL",
        "COMET-QE-MQM",
        "COMET-QE",
        "CometKiwi",
        "CometKiwi-XL",
        "GEMBA-MQM",
        "mbr-metricx-qe",
        "MaTESe-QE",
    ],
    "lexical based": ["eBLEU", "f200spBLEU", "BLEU", "chrF", "tokengram_F"],
    "sentinel metrics": ["SENTINEL-SRC-MQM", "SENTINEL-REF-MQM", "SENTINEL-CAND-MQM"],
}

mainpaper_metrics = [
    "XCOMET-Ensemble",
    "XCOMET-XL",
    "MetricX-23",
    "MetricX-23-XL",
    "MaTESe",
    "COMET",
    "BLEURT-20",
    "XCOMET-QE-Ensemble",
    "MetricX-23-QE",
    "MetricX-23-QE-XL",
    "COMET-QE-MQM",
    "COMET-QE",
    "CometKiwi",
    "CometKiwi-XL",
    "GEMBA-MQM",
    "mbr-metricx-qe",
    "MaTESe-QE",
]

lexical_metrics = ["BLEU", "chrF", "eBLEU", "f200spBLEU", "tokengram_F"]


def load_json_file(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
    return data


def adjust(value, metric, measure=False):
    if measure:
        value *= 100
    if isinstance(value, float):
        value = round(value, 2)
    value = f"{value:.2f}"
    return value


def write_latex_table(
    sys_scores_gold_1,
    sys_scores_gold_4,
    qe_reranking,
    dataset,
    output_file,
    appendix=False,
):
    with open(output_file, "w") as f:
        f.write("\\resizebox{\\linewidth}{!}{\n")
        f.write("    \\begin{NiceTabular}{ll|rrrr|rrrr|rr}[cell-space-limits=3pt]\n")
        f.write("    \\toprule\n")
        f.write(
            "     &  & \\multicolumn{4}{c|}{\\textbf{\\good vs \\bad}} & \\multicolumn{4}{c|}{\\textbf{\\perfect vs \\other}} & \\multicolumn{2}{c}{\\textbf{Re-ranking}} \\\\\n"
        )
        f.write(
            "    & \\textbf{Metric} & \\multicolumn{1}{c}{$\\boldsymbol{\\epsilon}$} & \\multicolumn{1}{c}{\\textbf{P}} & \\multicolumn{1}{c}{\\textbf{R}} & \\multicolumn{1}{c|}{$\\boldsymbol{F}$} & \\multicolumn{1}{c}{$\\boldsymbol{\\epsilon}$} & \\multicolumn{1}{c}{\\textbf{P}} & \\multicolumn{1}{c}{\\textbf{R}} & \\multicolumn{1}{c|}{$\\boldsymbol{F}$} & \\multicolumn{1}{c}{\\textbf{Acc.}} & \\multicolumn{1}{c}{\\textbf{Avg.}} \\\\\n"
        )

        for category, metrics in category2metrics.items():
            metrics = [metric for metric in metrics if metric in sys_scores_gold_1]
            metrics = [metric for metric in metric2command if metric in metrics]

            if not appendix:
                if category == "lexical based" or category == "sentinel metrics":
                    continue
                metrics = [metric for metric in metrics if metric in mainpaper_metrics]

            f.write("    \\cmidrule(lr){2-12}\n")

            f.write(
                f"    \\multirow{{ {len(metrics)} }}{{*}}{{\\rotatebox{{90}}{{\\shortstack{{\\textsc{{{category.split()[0]}}} \\\\ \\textsc{{{category.split()[1]}}}}}}}}}\n"
            )
            for metric in metrics:
                color = (
                    "\\cellcolor{highlight}"
                    if metric2highlight[metric] and dataset == "wmt23"
                    else ""
                )
                f.write(
                    f"    & {color}{metric2command[metric]} & {color}${adjust(sys_scores_gold_4[metric]['optimal_metric_threshold'], metric)}$ & {color}${adjust(sys_scores_gold_4[metric]['macro_precision'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_4[metric]['macro_recall'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_4[metric]['macro_f1'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_1[metric]['optimal_metric_threshold'], metric)}$ & {color}${adjust(sys_scores_gold_1[metric]['macro_precision'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_1[metric]['macro_recall'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_1[metric]['macro_f1'], metric, measure=True)}$ & {color}${adjust(qe_reranking[metric]['macro_qe_ranking_precision'], metric, measure=True)}$ & {color}${adjust(qe_reranking[metric]['overall_best_translation_metric_mqm_score_macro_avg'], metric)}$ \\\\\n"
                )

        color = ""
        metric = "Random-sysname"
        if sys_scores_gold_4.get(metric, None):
            f.write("    \\cmidrule(lr){2-12}\n")
            f.write(
                f"    & {color}{metric2command[metric]} & {color}${adjust(sys_scores_gold_4[metric]['optimal_metric_threshold'], metric)}$ & {color}${adjust(sys_scores_gold_4[metric]['macro_precision'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_4[metric]['macro_recall'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_4[metric]['macro_f1'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_1[metric]['optimal_metric_threshold'], metric)}$ & {color}${adjust(sys_scores_gold_1[metric]['macro_precision'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_1[metric]['macro_recall'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_1[metric]['macro_f1'], metric, measure=True)}$ & {color}${adjust(qe_reranking[metric]['macro_qe_ranking_precision'], metric, measure=True)}$ & {color}${adjust(qe_reranking[metric]['overall_best_translation_metric_mqm_score_macro_avg'], metric)}$ \\\\\n"
            )

        color = ""
        metric = "da-sqm"
        if sys_scores_gold_4.get(metric, None):
            if metric in sys_scores_gold_1:
                f.write(
                    f"    & {color}{metric2command[metric]} & {color}${adjust(sys_scores_gold_4[metric]['optimal_metric_threshold'], metric)}$ & {color}${adjust(sys_scores_gold_4[metric]['macro_precision'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_4[metric]['macro_recall'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_4[metric]['macro_f1'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_1[metric]['optimal_metric_threshold'], metric)}$ & {color}${adjust(sys_scores_gold_1[metric]['macro_precision'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_1[metric]['macro_recall'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_1[metric]['macro_f1'], metric, measure=True)}$ & {color}${adjust(qe_reranking[metric]['macro_qe_ranking_precision'], metric, measure=True)}$ & {color}${adjust(qe_reranking[metric]['overall_best_translation_metric_mqm_score_macro_avg'], metric)}$ \\\\\n"
                )

        f.write("    \\bottomrule\n")
        f.write("    \\end{NiceTabular}\n")
        f.write("}\n")


def write_latex_table_dev(
    sys_scores_gold_1,
    sys_scores_gold_4,
    dataset,
    output_file,
    appendix=False,
):
    with open(output_file, "w") as f:
        f.write("    \\begin{NiceTabular}{ll|rrrr|rrrr}[cell-space-limits=3pt]\n")
        f.write("    \\toprule\n")
        f.write(
            "     &  & \\multicolumn{4}{c|}{\\textbf{\\good vs \\bad}} & \\multicolumn{4}{c}{\\textbf{\\perfect vs \\other}} \\\\\n"
        )
        f.write(
            "    & \\textbf{Metric} & \\multicolumn{1}{c}{$\\boldsymbol{\\epsilon}$} & \\multicolumn{1}{c}{\\textbf{P}} & \\multicolumn{1}{c}{\\textbf{R}} & \\multicolumn{1}{c|}{$\\boldsymbol{F}$} & \\multicolumn{1}{c}{$\\boldsymbol{\\epsilon}$} & \\multicolumn{1}{c}{\\textbf{P}} & \\multicolumn{1}{c}{\\textbf{R}} & \\multicolumn{1}{c}{$\\boldsymbol{F}$} \\\\\n"
        )

        for category, metrics in category2metrics.items():
            metrics = [metric for metric in metrics if metric in sys_scores_gold_1]      

            metrics = [metric for metric in metrics if metric != 'XCOMET-XL']  
            
            if category == 'sentinel metrics':
                continue

            if not appendix:
                if category == "lexical based" or category == "sentinel metrics":
                    continue

                metrics = [metric for metric in metrics if metric in mainpaper_metrics]

            f.write("    \\cmidrule(lr){2-10}\n")

            f.write(
                f"    \\multirow{{ {len(metrics)} }}{{*}}{{\\rotatebox{{90}}{{\\small \\shortstack{{\\textsc{{{category.split()[0]}}} \\\\ \\textsc{{{category.split()[1]}}}}}}}}}\n"
            )
            for metric in metrics:


                color=""
                f.write(
                    f"    & {color}{metric2command[metric]} & {color}${adjust(sys_scores_gold_4[metric]['optimal_metric_threshold'], metric)}$ & {color}${adjust(sys_scores_gold_4[metric]['macro_precision'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_4[metric]['macro_recall'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_4[metric]['macro_f1'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_1[metric]['optimal_metric_threshold'], metric)}$ & {color}${adjust(sys_scores_gold_1[metric]['macro_precision'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_1[metric]['macro_recall'], metric, measure=True)}$ & {color}${adjust(sys_scores_gold_1[metric]['macro_f1'], metric, measure=True)}$ \\\\\n"
                )

        f.write("    \\bottomrule\n")
        f.write("    \\end{NiceTabular}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language-pair", "-lp", type=str, default="zh-en", help="Language pair"
    )
    parser.add_argument("--dataset", "-d", type=str, default="wmt23", help="Dataset")
    parser.add_argument(
        "--dev-set", "-dev", action="store_true", help="Use development set"
    )
    parser.add_argument("--apx", "-a", action="store_true", help="Appendix")
    args = parser.parse_args()

    lp = args.language_pair
    dataset = args.dataset
    apx = "-apx" if args.apx else ""

    if not args.dev_set:
        output_file = f"metrics_data/tex_artifacts/performance-results-{dataset}-{lp.split('-')[0]+lp.split('-')[1]}{apx}.tex"

        sys_scores_gold_1 = load_json_file(
            f"metrics_data/rankings/new_metrics/{dataset}/{lp}/f1/gold_score_threshold_-1.0__precision_threshold_0/sys_grouping_metrics_rankings.json"
        )
        sys_scores_gold_4 = load_json_file(
            f"metrics_data/rankings/new_metrics/{dataset}/{lp}/f1/gold_score_threshold_-4.0__precision_threshold_0/sys_grouping_metrics_rankings.json"
        )
        qe_reranking = load_json_file(
            f"metrics_data/rankings/new_metrics/{dataset}/{lp}/macro_qe_ranking_precision/ranking.json"
        )

        write_latex_table(
            sys_scores_gold_1,
            sys_scores_gold_4,
            qe_reranking,
            dataset,
            output_file,
            appendix=args.apx,
        )

    else:
        output_file = f"metrics_data/tex_artifacts/performance-results-{dataset}-{lp.split('-')[0]+lp.split('-')[1]}{apx}-dev.tex"

        sys_scores_gold_1 = load_json_file(
            f"metrics_data/rankings/new_metrics/{dataset}/{lp}/f1/gold_score_threshold_-1.0__precision_threshold_0/sys_grouping_metrics_rankings_wmt22_thresholds.json"
        )
        sys_scores_gold_4 = load_json_file(
            f"metrics_data/rankings/new_metrics/{dataset}/{lp}/f1/gold_score_threshold_-4.0__precision_threshold_0/sys_grouping_metrics_rankings_wmt22_thresholds.json"
        )

        write_latex_table_dev(
            sys_scores_gold_1,
            sys_scores_gold_4,
            dataset,
            output_file,
            appendix=args.apx,
        )
