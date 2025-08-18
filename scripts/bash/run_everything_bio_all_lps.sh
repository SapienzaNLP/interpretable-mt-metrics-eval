#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e



echo "Starting run_everything_bio_all_lps.sh script execution..."



# Execute Python scripts

#zh-en
python scripts/py/rank_metrics.py --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4 --testset-name bio
echo "zh-en bio item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --new-metrics --average-by sys --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4 --testset-name bio
echo "zh-en bio sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -1 --testset-name bio
echo "zh-en bio item-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --new-metrics --average-by sys --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -1 --testset-name bio
echo "zh-en bio sys-group gold_score_threshold=-1 finished."


echo "zh-en finished."


#en-de
python scripts/py/rank_metrics.py --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4 --testset-name bio --lp en-de
echo "en-de bio item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --new-metrics --average-by sys --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4 --testset-name bio --lp en-de
echo "en-de bio sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -1 --testset-name bio --lp en-de
echo "en-de bio item-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --new-metrics --average-by sys --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -1 --testset-name bio --lp en-de
echo "en-de bio sys-group gold_score_threshold=-1 finished."


echo "en-de finished."


#en-ru
python scripts/py/rank_metrics.py --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4 --testset-name bio --lp en-ru
echo "en-ru bio item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --new-metrics --average-by sys --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4 --testset-name bio --lp en-ru
echo "en-ru bio sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -1 --testset-name bio --lp en-ru
echo "en-ru bio item-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --new-metrics --average-by sys --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -1 --testset-name bio --lp en-ru
echo "en-ru bio sys-group gold_score_threshold=-1 finished."


echo "en-ru finished."




echo "All scripts executed successfully."
