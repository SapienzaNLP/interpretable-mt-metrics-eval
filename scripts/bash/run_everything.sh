#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e



echo "Starting run_everything.sh script execution..."



# Execute Python scripts

#zh-en
python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --precision-relative-weight 2.0
echo "wmt23 zh-en none-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --precision-relative-weight 2.0
echo "wmt23 zh-en none-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --precision-relative-weight 2.0
echo "wmt23 zh-en none-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by item --precision-relative-weight 2.0
echo "wmt23 zh-en item-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --average-by item --precision-relative-weight 2.0
echo "wmt23 zh-en item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --average-by item --precision-relative-weight 2.0
echo "wmt23 zh-en item-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --precision-relative-weight 2.0
echo "wmt23 zh-en sys-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --average-by sys --precision-relative-weight 2.0
echo "wmt23 zh-en sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --average-by sys --precision-relative-weight 2.0
echo "wmt23 zh-en sys-group gold_score_threshold=-1 finished."

echo "wmt23 zh-en finished."


python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 zh-en none-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 zh-en none-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 zh-en none-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by item --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 zh-en item-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --average-by item --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 zh-en item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --average-by item --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 zh-en item-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 zh-en sys-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --average-by sys --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 zh-en sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --average-by sys --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 zh-en sys-group gold_score_threshold=-1 finished."

echo "wmt22 zh-en finished."


echo "zh-en finished."



# en-de
python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --lp en-de --precision-relative-weight 2.0
echo "wmt23 en-de none-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --lp en-de --precision-relative-weight 2.0
echo "wmt23 en-de none-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --lp en-de --precision-relative-weight 2.0
echo "wmt23 en-de none-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by item --lp en-de --precision-relative-weight 2.0
echo "wmt23 en-de item-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --average-by item --lp en-de --precision-relative-weight 2.0
echo "wmt23 en-de item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --average-by item --lp en-de --precision-relative-weight 2.0
echo "wmt23 en-de item-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --lp en-de --precision-relative-weight 2.0
echo "wmt23 en-de sys-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --average-by sys --lp en-de --precision-relative-weight 2.0
echo "wmt23 en-de sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --average-by sys --lp en-de --precision-relative-weight 2.0
echo "wmt23 en-de sys-group gold_score_threshold=-1 finished."

echo "wmt23 en-de finished."


python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --lp en-de --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-de none-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --lp en-de --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-de none-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --lp en-de --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-de none-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by item --lp en-de --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-de item-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --average-by item --lp en-de --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-de item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --average-by item --lp en-de --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-de item-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --lp en-de --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-de sys-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --average-by sys --lp en-de --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-de sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --average-by sys --lp en-de --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-de sys-group gold_score_threshold=-1 finished."

echo "wmt22 en-de finished."


# he-en
python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --lp he-en --ref-to-use refB --precision-relative-weight 2.0
echo "wmt23 he-en none-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --lp he-en --ref-to-use refB --precision-relative-weight 2.0
echo "wmt23 he-en none-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --lp he-en --ref-to-use refB --precision-relative-weight 2.0
echo "wmt23 he-en none-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by item --lp he-en --ref-to-use refB --precision-relative-weight 2.0
echo "wmt23 he-en item-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --average-by item --lp he-en --ref-to-use refB --precision-relative-weight 2.0
echo "wmt23 he-en item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --average-by item --lp he-en --ref-to-use refB --precision-relative-weight 2.0
echo "wmt23 he-en item-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --lp he-en --ref-to-use refB --precision-relative-weight 2.0
echo "wmt23 he-en sys-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --average-by sys --lp he-en --ref-to-use refB --precision-relative-weight 2.0
echo "wmt23 he-en sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --average-by sys --lp he-en --ref-to-use refB --precision-relative-weight 2.0
echo "wmt23 he-en sys-group gold_score_threshold=-1 finished."

echo "he-en finished."



# en-ru
python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --lp en-ru --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-ru none-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --lp en-ru --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-ru none-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --lp en-ru --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-ru none-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by item --lp en-ru --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-ru item-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --average-by item --lp en-ru --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-ru item-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --average-by item --lp en-ru --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-ru item-group gold_score_threshold=-1 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by sys --lp en-ru --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-ru sys-group gold_score_threshold=0 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -4 --average-by sys --lp en-ru --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-ru sys-group gold_score_threshold=-4 finished."

python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --gold-score-threshold -1 --average-by sys --lp en-ru --precision-relative-weight 2.0 --testset-name wmt22
echo "wmt22 en-ru sys-group gold_score_threshold=-1 finished."

echo "en-ru finished."



echo "All scripts executed successfully."
