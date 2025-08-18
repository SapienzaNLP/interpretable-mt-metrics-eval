#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e



echo "Starting run_stability_analysis.sh script execution..."



# Execute Python scripts

#zh-en
python scripts/py/compute_and_save_stability_data.py --include-human --include-outliers --n-processes 12
echo "wmt23 zh-en gold_score_threshold=-1 finished."

python scripts/py/compute_and_save_stability_data.py --include-human --include-outliers --n-processes 12 --gold-score-threshold -4
echo "wmt23 zh-en gold_score_threshold=-4 finished."


echo "zh-en finished."


#en-de
python scripts/py/compute_and_save_stability_data.py --include-human --include-outliers --n-processes 12 --lp en-de
echo "wmt23 en-de gold_score_threshold=-1 finished."

python scripts/py/compute_and_save_stability_data.py --include-human --include-outliers --n-processes 12 --gold-score-threshold -4 --lp en-de
echo "wmt23 en-de gold_score_threshold=-4 finished."


echo "en-de finished."


#he-en
python scripts/py/compute_and_save_stability_data.py --include-human --include-outliers --n-processes 12 --lp he-en --ref-to-use refB
echo "wmt23 he-en gold_score_threshold=-1 finished."

python scripts/py/compute_and_save_stability_data.py --include-human --include-outliers --n-processes 12 --gold-score-threshold -4 --lp he-en --ref-to-use refB
echo "wmt23 he-en gold_score_threshold=-4 finished."


echo "he-en finished."



echo "All scripts executed successfully."