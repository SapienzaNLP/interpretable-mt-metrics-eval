#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e



echo "Starting test_precomputed_thresholds.sh script execution..."



# Execute Python scripts

#zh-en
python scripts/py/test_precomputed_thresholds.py --include-human --include-outliers --n-processes 12 --thresholds-from-pickle metrics_data/stability_analysis/new_metrics/wmt23/zh-en/gold_score_threshold_-1/sample_size2metrics_results.pickle --dev-set-name wmt23_subsample
echo "wmt23 zh-en gold_score_threshold=-1 finished."

python scripts/py/test_precomputed_thresholds.py --include-human --include-outliers --n-processes 12 --gold-score-threshold -4 --thresholds-from-pickle metrics_data/stability_analysis/new_metrics/wmt23/zh-en/gold_score_threshold_-4.0/sample_size2metrics_results.pickle --dev-set-name wmt23_subsample
echo "wmt23 zh-en gold_score_threshold=-4 finished."


echo "zh-en finished."


#en-de
python scripts/py/test_precomputed_thresholds.py --include-human --include-outliers --n-processes 12 --lp en-de --thresholds-from-pickle metrics_data/stability_analysis/new_metrics/wmt23/en-de/gold_score_threshold_-1/sample_size2metrics_results.pickle --dev-set-name wmt23_subsample
echo "wmt23 en-de gold_score_threshold=-1 finished."

python scripts/py/test_precomputed_thresholds.py --include-human --include-outliers --n-processes 12 --gold-score-threshold -4 --lp en-de --thresholds-from-pickle metrics_data/stability_analysis/new_metrics/wmt23/en-de/gold_score_threshold_-4.0/sample_size2metrics_results.pickle --dev-set-name wmt23_subsample
echo "wmt23 en-de gold_score_threshold=-4 finished."


echo "en-de finished."



echo "All scripts executed successfully."
