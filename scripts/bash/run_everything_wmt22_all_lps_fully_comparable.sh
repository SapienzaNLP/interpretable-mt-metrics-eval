#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e



echo "Starting run_everything_wmt22_all_lps_fully_comparable.sh script execution..."



# Execute Python scripts

#zh-en
python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4 --testset-name wmt22
echo "zh-en wmt22 item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4 --testset-name wmt22
echo "zh-en wmt22 sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -1 --testset-name wmt22
echo "zh-en wmt22 item-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -1 --testset-name wmt22
echo "zh-en wmt22 sys-group gold_score_threshold=-1 finished."

echo "zh-en finished."


#en-de
python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4 --lp en-de --testset-name wmt22
echo "en-de wmt22 item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4 --lp en-de --testset-name wmt22
echo "en-de wmt22 sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -1 --lp en-de --testset-name wmt22
echo "en-de wmt22 item-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -1 --lp en-de --testset-name wmt22
echo "en-de wmt22 sys-group gold_score_threshold=-1 finished."

echo "en-de finished."


#en-ru
python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4 --lp en-ru --testset-name wmt22
echo "en-ru wmt22 item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4 --lp en-ru --testset-name wmt22
echo "en-ru wmt22 sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -1 --lp en-ru --testset-name wmt22
echo "en-ru wmt22 item-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -1 --lp en-ru --testset-name wmt22
echo "en-ru wmt22 sys-group gold_score_threshold=-1 finished."

echo "en-ru finished."



echo "All scripts executed successfully."
