import json
import argparse
from scripts_utils import read_metric2command


metric2command = read_metric2command()

openly_available_reference_free_metrics_except_gemba = [
        "MetricX-23-QE-XL",
        "COMET-QE-MQM",
        "COMET-QE",
        "CometKiwi",
        "CometKiwi-XL",
        "MaTESe-QE",
]

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


def sort_ref_free(qe_reranking):
    reference_free_metrics = openly_available_reference_free_metrics_except_gemba

    reference_free_scores = [
        qe_reranking[metric]["macro_qe_ranking_precision"] for metric in reference_free_metrics
    ]

    metric_and_score = list(zip(reference_free_metrics, reference_free_scores))
    sorted_metrics, sorted_scores = zip(*sorted(metric_and_score, key=lambda x: x[1], reverse=True))
    return sorted_metrics, sorted_scores


def write_latex_table(
    mbr_reranking_zhen,
    qe_reranking_zhen,
    mbr_reranking_ende,
    qe_reranking_ende,
    mbr_reranking_heen,
    qe_reranking_heen,
    dataset,
    output_file,
    appendix=False,
):
    
    zhen_best_metric, zhen_second_best_metric = sort_ref_free(qe_reranking_zhen)[0][:2]
    ende_best_metric, ende_second_best_metric = sort_ref_free(qe_reranking_ende)[0][:2]
    heen_best_metric, heen_second_best_metric = sort_ref_free(qe_reranking_heen)[0][:2]

    zhen_best,  zhen_second_best = sort_ref_free(qe_reranking_zhen)[1][:2]
    ende_best,  ende_second_best = sort_ref_free(qe_reranking_ende)[1][:2]
    heen_best,  heen_second_best = sort_ref_free(qe_reranking_heen)[1][:2]

    print(f"Best metrics for {dataset} zh-en: {zhen_best_metric}, {zhen_second_best_metric}")
    print(f"Best metrics for {dataset} en-de: {ende_best_metric}, {ende_second_best_metric}")
    print(f"Best metrics for {dataset} he-en: {heen_best_metric}, {heen_second_best_metric}")

    with open(output_file, "w") as f:
        f.write("\\resizebox{\\columnwidth}{!}{\n")
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("& \\multicolumn{2}{c}{\\langpair{zh}{en}} &  \\multicolumn{2}{c}{\\langpair{en}{de}} &  \\multicolumn{2}{c}{\\langpair{he}{en}} \\\\ \n")
        f.write("\\textbf{Metric} & \\textbf{MBR} & \\textbf{Tab~\\ref{tab:performance-results-zhen}} &  \\textbf{MBR} & \\textbf{Tab~\\ref{tab:performance-results-ende-apx}} & \\textbf{MBR} & \\textbf{Tab~\\ref{tab:performance-results-heen-apx}} \\\\ \n")

        metrics = list(metric2command.keys())
        metrics = [metric for metric in metrics if metric in mbr_reranking_zhen]            

        f.write("\\midrule\n")

        for metric in metrics:

            f.write(
                f"{metric2command[metric]} & ${adjust(mbr_reranking_zhen[metric]['macro_mbr_precision'], metric, measure=True)}$ & ${adjust(qe_reranking_zhen[metric]['macro_qe_ranking_precision'], metric, measure=True)}$ & ${adjust(mbr_reranking_ende[metric]['macro_mbr_precision'], metric, measure=True)}$ & ${adjust(qe_reranking_ende[metric]['macro_qe_ranking_precision'], metric, measure=True)}$ & ${adjust(mbr_reranking_heen[metric]['macro_mbr_precision'], metric, measure=True)}$ & ${adjust(qe_reranking_heen[metric]['macro_qe_ranking_precision'], metric, measure=True)}$ \\\\\n"
            )
        
        f.write("\\cmidrule(lr){2-7}\n")

        f.write(f"\\#1 \\textsc{{ref free}} & -- & ${adjust(zhen_best, None, measure=True)}$ & -- & ${adjust(ende_best, None, measure=True)}$ & -- & ${adjust(heen_best, None, measure=True)}$ \\\\\n")
        f.write(f"\\#2 \\textsc{{ref free}} & -- & ${adjust(zhen_second_best, None, measure=True)}$ & -- & ${adjust(ende_second_best, None, measure=True)}$ & -- & ${adjust(heen_second_best, None, measure=True)}$ \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="wmt23", help="Dataset")
    parser.add_argument("--apx", "-a", action="store_true", help="Appendix")
    args = parser.parse_args()

    dataset = args.dataset
    apx = "-apx" if args.apx else ""

   
    output_file = f"metrics_data/tex_artifacts/mbr-results-{dataset}{apx}.tex"

    mbr_reranking_zhen = load_json_file(
        f"metrics_data/rankings/new_metrics/{dataset}/zh-en/macro_mbr_precision/ranking.json"
    )
    qe_reranking_zhen = load_json_file(
        f"metrics_data/rankings/new_metrics/{dataset}/zh-en/macro_qe_ranking_precision/ranking.json"
    )
    mbr_reranking_ende = load_json_file(
        f"metrics_data/rankings/new_metrics/{dataset}/en-de/macro_mbr_precision/ranking.json"
    )
    qe_reranking_ende = load_json_file(
        f"metrics_data/rankings/new_metrics/{dataset}/en-de/macro_qe_ranking_precision/ranking.json"
    )
    mbr_reranking_heen = load_json_file(
        f"metrics_data/rankings/new_metrics/{dataset}/he-en/macro_mbr_precision/ranking.json"
    )
    qe_reranking_heen = load_json_file(
        f"metrics_data/rankings/new_metrics/{dataset}/he-en/macro_qe_ranking_precision/ranking.json"
    )


    write_latex_table(
        mbr_reranking_zhen,
        qe_reranking_zhen,
        mbr_reranking_ende,
        qe_reranking_ende,
        mbr_reranking_heen,
        qe_reranking_heen,
        dataset,
        output_file,
        appendix=args.apx,
    )
