#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e



echo "Starting run_item_wmt23_all_lps_fully_comparable.sh script execution..."



# Execute Python scripts

#zh-en
python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4
echo "zh-en wmt23 item-group finished."


#en-de
python scripts/py/rank_metrics.py --include-human --include-outliers --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4 --lp en-de
echo "en-de wmt23 item-group finished."


#he-en
python scripts/py/rank_metrics.py --ref-to-use refB --include-human --include-outliers --new-metrics --average-by item --precision-relative-weight 2.0 --n-processes 12 --gold-score-threshold -4 --lp he-en
echo "he-en wmt23 item-group finished."



echo "All scripts executed successfully."
