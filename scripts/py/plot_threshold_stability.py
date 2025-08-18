import json
import matplotlib.pyplot as plt
from scripts_utils import read_metric2latex

def load_json_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


datasets = {
    'wmt22': ['zh-en', 'en-de', 'en-ru'],
    'wmt23': ['zh-en', 'en-de', 'he-en'],
}

def load_all_data():
    all_sys_scores_gold_1 = {}
    for dataset, lps in datasets.items():
        all_sys_scores_gold_1[dataset] = {}
        for lp in lps:
            sys_scores_gold_1_tmp = load_json_file(f"rankings/new_metrics/{dataset}/{lp}/f1/gold_score_threshold_-1.0__precision_threshold_0/sys_grouping_metrics_rankings.json")
            all_sys_scores_gold_1[dataset][lp] = sys_scores_gold_1_tmp
    
    return all_sys_scores_gold_1


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.sans-serif": ["Computer Modern Sans"],
    "font.monospace": ["Computer Modern Typewriter"],
    "axes.unicode_minus": False  # Avoid using Unicode minus
})

dataset2latex = {
    'wmt23-zhen': "\\textsc{zh}$\\rightarrow$\\textsc{en}",
    'wmt23-ende': "\\textsc{en}$\\rightarrow$\\textsc{de}",
    'wmt23-heen': "\\textsc{he}$\\rightarrow$\\textsc{en}",

}

metric2latex = read_metric2latex()


metrics2minmax = {
    'XCOMET-Ensemble': (0.17247988249433896, 0.996026849327445),
    'XCOMET-QE-Ensemble': (0.0962455979161886, 1.0),
    'MetricX-23': (-25.618242263793945, 0.363231360912323),
    'MetricX-23-QE': (-24.55722427368164, 0.9240800142288208),
    'GEMBA-MQM': (-25.0, 0.0),
    'mbr-metricx-qe': (-0.00398574, 0.999841),
    'MaTESe': (-25.0, 0.0),
    'XCOMET-XL': (-0.026581795886158943, 1.0),
    'MetricX-23-XL': (-25.0, -0.0),
    'MetricX-23-QE-XL': (-25.0, -0.0),
    'CometKiwi-XL': (-0.15926963090896606, 0.9072170853614807),
    'CometKiwi': (0.19868940114974976, 0.9025614261627197),
    'COMET': (0.1772385537624359, 0.9948719143867493),
    'BLEURT-20': (0.00016555190086364746, 1.0825953483581543),
    'COMET-QE': (-0.6250770688056946, 0.2970536947250366),
    'COMET-QE-MQM': (-0.04704536870121956, 0.18359123170375824),
    'SENTINEL-CAND': (-2.8625268936157227, 0.9339103102684021),
    'SENTINEL-SRC': (-2.6017770767211914, 0.9515202641487122),
    'SENTINEL-REF': (-2.693288803100586, 1.0216286182403564),
    'MaTESe-QE': (-25, 0),
    'Random-sysname': (-7.0, 16.0)
}

def normalize_threshold(threshold, metric):
    if metric in metrics2minmax:
        minval, maxval = metrics2minmax[metric]
        threshold = (threshold - minval) / (maxval - minval)
    else:
        print(f"Unknown metric: {metric}")

    return threshold



def plot_statistic(all_sys_scores_gold_1, statistic, title, xlabel, ylabel):
    metric2thresholds = {metric: [] for metric in all_metrics}
    ordereddatasets = ['wmt23-zhen', 'wmt23-ende', 'wmt23-heen']
    for metric in all_metrics:
        for dataset in datasets:
            for lp in datasets[dataset]:
                if dataset.endswith('22'):
                    continue
                sys_scores_gold_1 = all_sys_scores_gold_1[dataset][lp]
                threshold = normalize_threshold(sys_scores_gold_1[metric][statistic], metric)
                metric2thresholds[metric].append(threshold)
    
    ordereddatasets = [dataset2latex[dataset] for dataset in ordereddatasets]
    plt.figure(figsize=(12, 8))

    for metric in all_metrics:
        plt.plot(ordereddatasets, metric2thresholds[metric], marker='o', label=metric2latex[metric])

    plt.ylabel('Threshold $\\epsilon$', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    plt.grid(True)
    plt.savefig("thresholds-full-stability.pdf", format='pdf')



def plot_ranking(all_sys_scores_gold_1):
    
    ordereddatasets = ['wmt23-zhen', 'wmt23-ende', 'wmt23-heen']
    dataset2metrics = {dataset: [] for dataset in ordereddatasets}
    for dataset in datasets:
        for lp in datasets[dataset]:
            sys_scores_gold_1 = all_sys_scores_gold_1[dataset][lp]
            metrics = [(metric, sys_scores_gold_1[metric]['macro_f1'])  for metric in sys_scores_gold_1.keys() if metric in open_metrics]
            metrics = [metric for metric, _ in sorted(metrics, key=lambda x: x[1], reverse=True)]
            dataset2metrics[f"{dataset}-{lp.split('-')[0]+lp.split('-')[1]}"] = metrics
    
    sorted_metrics = sorted(open_metrics)

    rankings = {
        dataset: [dataset2metrics[dataset].index(metric) + 1 for metric in sorted_metrics] 
        for dataset in ordereddatasets
    }

    colors = plt.cm.get_cmap('tab10', len(sorted_metrics))

    # Plotting
    plt.figure(figsize=(10, 6))
    for idx, metric in enumerate(sorted_metrics):
        metric_ranks = [rankings[dataset][idx] for dataset in ordereddatasets]
        plt.plot(ordereddatasets, metric_ranks, marker='o', linestyle='-', label=metric, color=colors(idx))

    plt.xlabel('Datasets')
    plt.ylabel('Ranks')
    plt.title('Metric Rankings by Dataset')
    plt.gca().invert_yaxis()  # Invert y-axis so that lower ranks are at the top
    plt.legend(title='Metrics')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    
    all_sys_scores_gold_1 = load_all_data()

    plot_statistic(all_sys_scores_gold_1, 'optimal_metric_threshold', title="", xlabel="Datasets", ylabel="Threshold Values")
    #plot_statistic(all_sys_scores_gold_1, 'macro_f1')

    #plot_ranking(all_sys_scores_gold_1)

   

