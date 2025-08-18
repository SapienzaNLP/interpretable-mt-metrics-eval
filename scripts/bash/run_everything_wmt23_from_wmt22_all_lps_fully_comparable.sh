#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e



echo "Starting run_everything_wmt23_from_wmt22_all_lps_fully_comparable.sh script execution..."



# Execute Python scripts

#zh-en
python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --n-processes 12 --gold-score-threshold -4 --thresholds-from-json metrics_data/rankings/new_metrics/wmt22/zh-en/f1/gold_score_threshold_-4.0__precision_threshold_0/sys_grouping_metrics_rankings.json
echo "zh-en wmt23 sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --n-processes 12 --gold-score-threshold -1 --thresholds-from-json metrics_data/rankings/new_metrics/wmt22/zh-en/f1/gold_score_threshold_-1.0__precision_threshold_0/sys_grouping_metrics_rankings.json
echo "zh-en wmt23 sys-group gold_score_threshold=-1 finished."

echo "zh-en finished."


#en-de
python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --n-processes 12 --gold-score-threshold -4 --lp en-de --thresholds-from-json metrics_data/rankings/new_metrics/wmt22/en-de/f1/gold_score_threshold_-4.0__precision_threshold_0/sys_grouping_metrics_rankings.json
echo "en-de wmt23 sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --n-processes 12 --gold-score-threshold -1 --lp en-de --thresholds-from-json metrics_data/rankings/new_metrics/wmt22/en-de/f1/gold_score_threshold_-1.0__precision_threshold_0/sys_grouping_metrics_rankings.json
echo "en-de wmt23 sys-group gold_score_threshold=-1 finished."

echo "en-de finished."


echo "All scripts executed successfully."
