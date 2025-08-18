#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e



echo "Starting run_sys_wmt22_all_lps_fully_comparable.sh script execution..."



# Execute Python scripts

#zh-en
python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --n-processes 12 --gold-score-threshold -4 --testset-name wmt22
echo "zh-en wmt22 sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --n-processes 12 --gold-score-threshold -1 --testset-name wmt22
echo "zh-en wmt22 sys-group gold_score_threshold=-1 finished."

echo "zh-en finished."


#en-de
python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --n-processes 12 --gold-score-threshold -4 --lp en-de --testset-name wmt22
echo "en-de wmt22 sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --n-processes 12 --gold-score-threshold -1 --lp en-de --testset-name wmt22
echo "en-de wmt22 sys-group gold_score_threshold=-1 finished."

echo "en-de finished."



echo "All scripts executed successfully."
