#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e



echo "Starting run_everything_wmt22+23.sh script execution..."



# Execute Python scripts

#zh-en
python scripts/py/rank_metrics.py --include-human --include-outliers --include-ref-to-use --new-metrics --average-by item --testset-name all --precision-relative-weight 2.0
echo "wmt22+23 zh-en item-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --include-ref-to-use --new-metrics --gold-score-threshold -4 --average-by item --testset-name all --precision-relative-weight 2.0
echo "wmt22+23 zh-en item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --include-ref-to-use --new-metrics --average-by sys --testset-name all --precision-relative-weight 2.0
echo "wmt22+23 zh-en sys-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --include-ref-to-use --new-metrics --gold-score-threshold -4 --average-by sys --testset-name all --precision-relative-weight 2.0
echo "wmt22+23 zh-en sys-group gold_score_threshold=-4 finished."

echo "zh-en finished."


# en-de
python scripts/py/rank_metrics.py --include-human --include-outliers --include-ref-to-use --new-metrics --average-by item --lp en-de --testset-name all --precision-relative-weight 2.0
echo "wmt22+23 en-de item-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --include-ref-to-use --new-metrics --gold-score-threshold -4 --average-by item --lp en-de --testset-name all --precision-relative-weight 2.0
echo "wmt22+23 en-de item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --include-ref-to-use --new-metrics --average-by sys --lp en-de --testset-name all --precision-relative-weight 2.0
echo "wmt22+23 en-de sys-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --include-ref-to-use --new-metrics --gold-score-threshold -4 --average-by sys --lp en-de --testset-name all --precision-relative-weight 2.0
echo "wmt22+23 en-de sys-group gold_score_threshold=-4 finished."

echo "en-de finished."


# en-ru
python scripts/py/rank_metrics.py --include-human --include-outliers --include-ref-to-use --new-metrics --average-by item --lp en-ru --testset-name all --precision-relative-weight 2.0
echo "wmt22+23 en-ru item-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --include-ref-to-use --new-metrics --gold-score-threshold -4 --average-by item --lp en-ru --testset-name all --precision-relative-weight 2.0
echo "wmt22+23 en-ru item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --include-ref-to-use --new-metrics --average-by sys --lp en-ru --testset-name all --precision-relative-weight 2.0
echo "wmt22+23 en-ru sys-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --include-ref-to-use --new-metrics --gold-score-threshold -4 --average-by sys --lp en-ru --testset-name all --precision-relative-weight 2.0
echo "wmt22+23 en-ru sys-group gold_score_threshold=-4 finished."

echo "en-ru finished."



echo "All scripts executed successfully."
