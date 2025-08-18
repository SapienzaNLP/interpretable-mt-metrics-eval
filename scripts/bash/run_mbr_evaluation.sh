#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e


echo "Starting run_mbr_evaluation.sh script execution..."


# Execute Python scripts

#zh-en
python scripts/py/rank_metrics_mbr.py --include-human --include-outliers --new-metrics
echo "zh-en finished."

#en-de
python scripts/py/rank_metrics_mbr.py --include-human --include-outliers --new-metrics --lp en-de
echo "en-de finished."

#he-en
python scripts/py/rank_metrics_mbr.py --include-human --include-outliers --new-metrics --lp he-en --ref-to-use refB
echo "he-en finished."


echo "All scripts executed successfully."