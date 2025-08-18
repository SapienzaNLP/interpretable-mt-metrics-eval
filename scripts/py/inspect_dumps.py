from pathlib import Path
from typing import List, Dict, Any
import logging
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import numpy as np
import seaborn as sns
from scipy.stats import linregress

logging.basicConfig(level=logging.INFO)

zhen_dumps = Path('metrics_data/dumps/wmt23/zh-en')
ende_dumps = Path('metrics_data/dumps/wmt23/en-de')
heen_dumps = Path('metrics_data/dumps/wmt23/he-en')

gold_thresholds = ['0', '-1.0', '-2.0', '-3.0']
precision_thresholds = ['0']

plt.rcParams['text.usetex'] = True


def NER(data: List[str], batch_size=128, nerdir=Path('metrics_data/biases/ner_counts'), lp='zh-en', dataset='wmt23'):
    if not nerdir.exists():
        nerdir.mkdir(parents=True)
    
    filepath = nerdir / f'ner_counts.{dataset}.{lp}.pkl'

    if not filepath.exists():
        tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
        model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner").to('cuda')
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

        counts = []
        for idx in tqdm(range(0, len(data), batch_size), total=len(data)//batch_size):
            mts = [s['mt'] for s in data[idx:idx+batch_size]]
            srcs = [s['src'] for s in data[idx:idx+batch_size]]

            batch_result = nlp(mts)
        
            counts.extend(
                [{
                    'segment': srcs[idx],
                    'count': len(sample) 
                } for idx, sample in enumerate(batch_result)]
            )

        pickle.dump(counts, open(filepath, 'wb'))
    else:
        counts = pickle.load(open(filepath, 'rb'))
    
    return counts



class Data:
    def __init__(self, data: List[Dict], lp: str, dataset: str, optim_log_type=None):
        self.data = data

        self.lp = lp
        self.dataset = dataset
        self.optim_log_type = optim_log_type
        
        self.metric_name = self.data[0]['metric_name']
    
    def add_statistic(self, stat_name: str):
        if stat_name == "Candidate length":
            for sample in self.data:
                sample[stat_name] = len(sample['mt'])
        
        elif stat_name == "Reference length":
            for sample in self.data:
                sample[stat_name] = len(sample['ref'])
        
        elif stat_name == "Source length":
            for sample in self.data:
                sample[stat_name] = len(sample['src'])
            
        elif stat_name == "Candidate NE count":
            counts = NER(self.data, lp=self.lp, dataset=self.dataset)
            for idx, sample in enumerate(self.data):
                sample[stat_name] = counts[idx]['count'] / len(sample['mt'].split())
            
        else:
            raise NotImplementedError(f"Statistic {stat_name} not implemented")

    @classmethod
    def load(cls, filepath: Path, lp: str, dataset: str):
        if not filepath.exists() or filepath.is_dir():
            raise FileNotFoundError(f"File {filepath} does not exist or is a directory")

        with open(filepath, 'r') as fin:
            lines = fin.readlines()


        data = []

        idx = 0
        while idx < len(lines):
            src = lines[idx].strip()[len('SRC: '):].strip()
            sys_name = lines[idx+1].split(' ')[0].strip()
            mt = lines[idx+1][len(sys_name)+len(' CAND: '):].strip()
            ref = lines[idx+2][len('REF: '):].strip()
            gold_score = float(lines[idx+3][len("GOLD mqm score: "):])
            metric_name = lines[idx+4].split(' ')[0].strip()
            score = float(lines[idx+4][len(metric_name)+len(' score: '):].strip())      

            logging.debug(f"src: {src}")
            logging.debug(f"sys_name: {sys_name}")
            logging.debug(f"mt: {mt}")
            logging.debug(f"ref: {ref}")
            logging.debug(f"metric_name: {metric_name}")
            logging.debug(f"score: {score}")

            data.append(
                {
                    "src": src,
                    "sys_name": sys_name,
                    "mt": mt,
                    "ref": ref,
                    "gold_score": gold_score,
                    "metric_name": metric_name,
                    "score": score
                }
            )

            idx += 8
        
        return cls(data, lp=lp, dataset=dataset)
    
    @classmethod
    def load_optim_log(cls, filepath: Path, lp: str, dataset: str, metric_name: str):
        if not filepath.exists() or filepath.is_dir():
            raise FileNotFoundError(f"File {filepath} does not exist or is a directory")

        with open(filepath, 'r') as fin:
            lines = fin.readlines()

        data = []

        idx = 0
        while idx < len(lines):
            src = lines[idx].strip()[len('SRC: '):].strip()
            mt = lines[idx+1][len('CAND: '):].strip()
            ref = lines[idx+2][len('REF: '):].strip()
            score = float(lines[idx+3][len('METRIC SCORE: '):].strip())  
            gold_score = float(lines[idx+4][len("HUMAN SCORE: "):].strip()) 

            logging.debug(f"src: {src}")
            logging.debug(f"mt: {mt}")
            logging.debug(f"ref: {ref}")
            logging.debug(f"score: {score}")
            logging.debug(f"gold_score: {gold_score}")

            data.append(
                {
                    "src": src,
                    "mt": mt,
                    "ref": ref,
                    "gold_score": gold_score,
                    "metric_name": metric_name,
                    "score": score
                }
            )

            idx += 7
        
        return cls(data, lp=lp, dataset=dataset, optim_log_type=filepath.stem)
    
    @classmethod
    def cat(cls, o1, o2, o3):
        lp = o1.lp
        if o1.lp != o2.lp or o1.lp != o3.lp:
            logging.warning(f"You are concatenating data from different language pairs: {o1.lp} and {o2.lp} and {o3.lp}")
            lp = 'all'

        dataset = o1.dataset        
        if o1.dataset != o2.dataset or o1.dataset != o3.dataset:
            logging.warning(f"You are concatenating data from different datasets: {o1.dataset} and {o2.dataset} and {o3.dataset}")
            dataset = 'all'
        
        return cls(o1.data + o2.data + o3.data, lp, dataset)
    
    def group_by_segment(self):
        segment_grouping = {}
        for sample in self.data:
            segment = sample['src']
            segment_grouping[segment] = segment_grouping.get(segment, [])
            segment_grouping[segment].append(sample)

        grouped_data = []
        for segment, group in segment_grouping.items():
            grouped_data.append(
                {
                    "src": segment,
                    "ref": group[0]['ref'],
                    "samples": group,
                    "gold_score": sum([sample['gold_score'] for sample in group]) / len(group),
                    "metric_name": group[0]['metric_name'],
                    "score": sum([sample['score'] for sample in group]) / len(group)
                }
            )
        
        return grouped_data
    

class Datastats:
    statistics = [
        "Candidate length",
        "Reference length",
        "Source length",
        #"Candidate NE count"
    ]
    metric_name2title = {
        'XCOMET-Ensemble': 'XCOMET-Ensemble',
        'XCOMET-QE-Enseble': 'XCOMET-QE-Ensemble',
        'MQM': 'MQM',
        'SRC-ONLY-FAKE-METRIC-MQM': r'\textsc{sentinel}$_{\textsc{src}}$',
        'REF-ONLY-FAKE-METRIC-MQM': r'\textsc{sentinel}$_{\textsc{ref}}$',
        'CAND-ONLY-FAKE-METRIC-MQM': r'\textsc{sentinel}$_{\textsc{cand}}$',
    }

    def __init__(self, data: Data, lp: str, dataset: str, **kwargs):
        self.data = data
        self.lp = lp
        self.metric_name = data.metric_name

        if self.data.optim_log_type is None:
            self.savedir = Path(f'metrics_data/biases/{dataset}/{lp}')
        else:
            gold_score_threshold = kwargs['gold_score_threshold']
            precision_threshold = kwargs['precision_threshold']
            self.savedir = Path(f'metrics_data/biases/{dataset}/{lp}/optim_logs/{self.metric_name}/gold_score_threshold_{gold_score_threshold}__precision_threshold_{precision_threshold}/{self.data.optim_log_type}')
        if not self.savedir.exists():
            self.savedir.mkdir(parents=True)
        
        for statistic in self.statistics:
            self.data.add_statistic(statistic)
    
    def plot_all_stats(self, plot_mqm=False):
        for stat in self.statistics:
            if plot_mqm:
                self.plot_statistic(stat, metric=False)
            self.plot_statistic(stat, metric=True)
    
    def plot_combined_plot(self, stat_name: str):
        data = self.data.data
        datadir = self.savedir / stat_name

        if not datadir.exists():
            datadir.mkdir(parents=True)

        def normalize(data):
            data_min = np.min(data)
            data_max = np.max(data)
            return (data - data_min) / (data_max - data_min)

        all_x = [sample[stat_name] for sample in data]
        all_y = [sample['score'] for sample in data]
        
        all_gold_x  = all_x
        all_gold_y = [sample['gold_score'] for sample in data]

        # remove outliers from the gold MQM scores (there are some scores that are lower than -25)
        gold_x, gold_y = self.remove_mqm_wrong_values(all_gold_x, all_gold_y)
        x, _ = self.remove_mqm_wrong_values(all_x, all_gold_y)
        y, _ = self.remove_mqm_wrong_values(all_y, all_gold_y)
        
        gold_x, gold_y = self.remove_x_outliers(gold_x, gold_y)
        x, y = self.remove_x_outliers(x,y)

        assert(len(x) == len(y) == len(gold_x) == len(gold_y)), f"Lengths are not equal: {len(x)}, {len(y)}, {len(gold_x)}, {len(gold_y)}"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

        ax1.scatter(x, y, alpha=0.6, edgecolors='w', linewidth=0.5)
        ax1.grid(True, linestyle='--', alpha=0.6)
        slope1, intercept1, r_value, p_value, std_err = linregress(x, y)
        line1 = slope1 * np.array(x) + intercept1
        ax1.plot(x, line1, color='r')
        camera_ready_metric_name = self.metric_name2title.get(self.metric_name, self.metric_name)
        ax1.set_ylabel(f'{camera_ready_metric_name} score', fontsize=16)

        ax2.scatter(gold_x, gold_y, alpha=0.6, edgecolors='w', linewidth=0.5)
        ax2.grid(True, linestyle='--', alpha=0.6)
        slope2, intercept2, r_value, p_value, std_err = linregress(gold_x, gold_y)
        line2 = slope2 * np.array(gold_x) + intercept2
        ax2.plot(gold_x, line2, color='r')
        ax2.set_xlabel(stat_name, fontsize=16)
        ax2.set_ylabel(f'MQM score', fontsize=16)

        fig.tight_layout()

        #ax2.hist2d(x, y, bins=25)

        # Save the plot
        plt_name = f"combined_{stat_name}.png"
        plt.savefig(datadir / plt_name, format='png')
        plt.savefig(datadir / plt_name.replace('.png', '.pdf'), format='pdf')
        plt.close()
    

    def plot_statistic(self, stat_name: str, metric: bool):
        
        data = self.data.data
        datadir = self.savedir / stat_name

        if not datadir.exists():
            datadir.mkdir(parents=True)

        def normalize(data):
            data_min = np.min(data)
            data_max = np.max(data)
            return (data - data_min) / (data_max - data_min)

        metric_name = self.metric_name if metric else "MQM"
        x = [sample[stat_name] for sample in data]
        if metric:
            y = [sample['score'] for sample in data]
        else:
            y = [sample['gold_score'] for sample in data]
            # remove outliers from the gold MQM scores (there are some scores that are too low, e.g. -50)
            # x, y = self.remove_mqm_wrong_values(x,y)
        
        x, y = self.remove_x_outliers(x,y)

        # Create figure and axis objects
        fig, ax = plt.subplots(figsize=(6, 4), dpi=72)

        # Scatter plot with transparency and different marker
        scatter = ax.scatter(x, y, alpha=0.6, edgecolors='w', linewidth=0.5, rasterized=True)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)

        # Normalize x and y values
        # x_normalized = normalize(np.array(x))
        # y_normalized = normalize(np.array(y))
        
        # Perform linear regression on normalized data
        # slope_normalized, intercept_normalized, r_value, p_value, std_err = linregress(x_normalized, y_normalized)

        # Add regression line
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        line = slope * np.array(x) + intercept

        #ax.plot(x, line, color='r', label=f'Trend line (slope={slope_normalized:.2f})')
        ax.plot(x, line, color='r')
        
        camera_ready_metric_name = self.metric_name2title.get(metric_name, metric_name)
        # Set axis labels and title
        ax.set_xlabel(stat_name, fontsize=16)
        ax.set_ylabel(f'{camera_ready_metric_name} score', fontsize=16)
        #ax.set_title(f'{title}', fontsize=20)

        fig.tight_layout()

        #ax2.hist2d(x, y, bins=25)

        # Save the plot
        plt_name = f"{metric_name}_{stat_name}.png"
        plt.savefig(datadir / plt_name, format='png')
        plt.savefig(datadir / plt_name.replace('.png', '.pdf'), format='pdf')
        plt.close()
    
    def remove_mqm_wrong_values(self, x, y):
        allowed = [-25, 0]
        x = [x[idx] for idx in range(len(y)) if y[idx] >= allowed[0] and y[idx] <= allowed[1]]
        y = [y[idx] for idx in range(len(y)) if y[idx] >= allowed[0] and y[idx] <= allowed[1]]

        return x, y
    
    def remove_x_outliers(self, x, y):
        outliers = np.percentile(x, [0.5, 99.5])
        # keep equality so you don't remove 0s in MQM, for example
        y = [y[idx] for idx in range(len(x)) if x[idx] >= outliers[0] and x[idx] <= outliers[1]]
        x = [x[idx] for idx in range(len(x)) if x[idx] >= outliers[0] and x[idx] <= outliers[1]]

        return x,y


if __name__ == "__main__":
    sorted_zhen_dumps = sorted(zhen_dumps.iterdir())
    sorted_ende_dumps = sorted(ende_dumps.iterdir())
    sorted_heen_dumps = sorted(heen_dumps.iterdir())
    for idx, (zhen_dump, ende_dump, heen_dump) in enumerate(zip(sorted_zhen_dumps, sorted_ende_dumps, sorted_heen_dumps)):

        if idx == 0:
            plot_mqm = True
        else:
            plot_mqm = False
        
        assert(zhen_dump.stem == ende_dump.stem == heen_dump.stem), f"Dumps are not alinged: {zhen_dump.stem}, {ende_dump.stem}, {heen_dump.stem}"

        if any([dump.is_dir() for dump in [zhen_dump, ende_dump, heen_dump]]):
            continue
        
        zhen_lp = str(zhen_dump).split('/')[-2]
        zhen_dataset = str(zhen_dump).split('/')[-3]
        ende_lp = str(ende_dump).split('/')[-2]
        ende_dataset = str(ende_dump).split('/')[-3]
        heen_lp = str(heen_dump).split('/')[-2]
        heen_dataset = str(heen_dump).split('/')[-3]

        # concatenate data from all language pairs
        zhen_data = Data.load(zhen_dump, lp=zhen_lp, dataset=zhen_dataset)
        ende_data = Data.load(ende_dump, lp=ende_lp, dataset=ende_dataset)
        heen_data = Data.load(heen_dump, lp=heen_lp, dataset=heen_dataset)
        data = Data.cat(zhen_data, ende_data, heen_data)

        logging.info(f"Processing All language pairs for dataset: {zhen_dataset} and metric: {data.metric_name}")

        zhen_stats = Datastats(zhen_data, lp=zhen_lp, dataset=zhen_dataset)
        zhen_stats.plot_all_stats(plot_mqm=plot_mqm)
        ende_stats = Datastats(ende_data, lp=ende_lp, dataset=ende_dataset)
        ende_stats.plot_all_stats(plot_mqm=plot_mqm)
        heen_stats = Datastats(heen_data, lp=heen_lp, dataset=heen_dataset)
        heen_stats.plot_all_stats(plot_mqm=plot_mqm)

        stats = Datastats(data, lp=data.lp, dataset=data.dataset)
        stats.plot_all_stats(plot_mqm=plot_mqm)

        if zhen_data.metric_name == 'XCOMET-Ensemble':
            zhen_stats = Datastats(zhen_data, lp=zhen_lp, dataset=zhen_dataset)
            for stat in Datastats.statistics:
                zhen_stats.plot_combined_plot(stat)


    """
    
    optim_logs = [dump for dump in all_dumps if dump.is_dir()]

    for optim_log in optim_logs:
        for gold_score_threshold in gold_thresholds:
            for precision_threshold in precision_thresholds:
                lp = str(optim_log).split('/')[-2]
                dataset = str(optim_log).split('/')[-3]

                dirpath = optim_log / f"gold_score_threshold_{gold_score_threshold}__precision_threshold_{precision_threshold}"
                if not dirpath.exists():
                    logging.warning(f"Dirpath {dirpath} does not exist, skipping it.")
                    continue
                                
                for metric in dirpath.iterdir():
                    for optim_log_dump in metric.iterdir():
                        logging.info(f"Processing dataset: {dataset}\tLP: {lp}\tMetric: {metric.stem}\tGold threshold: {gold_score_threshold}\tPrecision threshold: {precision_threshold}\tType:{optim_log_dump.stem}")
                
                        data = Data.load_optim_log(optim_log_dump, lp=lp, dataset=dataset, metric_name=metric.stem)
                        stats = Datastats(data, lp=lp, dataset=dataset, gold_score_threshold=gold_score_threshold, precision_threshold=precision_threshold)
                        stats.plot_all_stats()
    
    """
