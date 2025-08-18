#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e



echo "Starting run_everything_wmt22+23_qe_ranking.sh script execution..."



# Execute Python scripts

#zh-en
python scripts/py/rank_metrics.py --include-outliers --new-metrics --average-by item --testset-name all --precision-relative-weight 2.0 --n-processes 12
echo "zh-en finished."


# en-de
python scripts/py/rank_metrics.py --include-outliers --new-metrics --average-by item --lp en-de --testset-name all --precision-relative-weight 2.0 --n-processes 12
echo "en-de finished."


# en-ru
python scripts/py/rank_metrics.py --include-outliers --new-metrics --average-by item --lp en-ru --testset-name wmt22 --precision-relative-weight 2.0 --n-processes 12
echo "en-ru finished."



echo "All scripts executed successfully."
