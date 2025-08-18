#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e



echo "Starting run_stability_analysis.sh script execution..."



# Execute Python scripts

#zh-en
python scripts/py/compute_and_save_stability_data.py --include-human --include-outliers --n-processes 12 --testset-name wmt22
echo "wmt22 zh-en gold_score_threshold=-1 finished."

python scripts/py/compute_and_save_stability_data.py --include-human --include-outliers --n-processes 12 --gold-score-threshold -4 --testset-name wmt22
echo "wmt22 zh-en gold_score_threshold=-4 finished."


echo "zh-en finished."


#en-de
python scripts/py/compute_and_save_stability_data.py --include-human --include-outliers --n-processes 12 --lp en-de --testset-name wmt22
echo "wmt22 en-de gold_score_threshold=-1 finished."

python scripts/py/compute_and_save_stability_data.py --include-human --include-outliers --n-processes 12 --gold-score-threshold -4 --lp en-de --testset-name wmt22
echo "wmt22 en-de gold_score_threshold=-4 finished."


echo "en-de finished."


#en-ru
python scripts/py/compute_and_save_stability_data.py --include-human --include-outliers --n-processes 12 --lp en-ru --testset-name wmt22
echo "wmt22 en-ru gold_score_threshold=-1 finished."

python scripts/py/compute_and_save_stability_data.py --include-human --include-outliers --n-processes 12 --gold-score-threshold -4 --lp en-ru --testset-name wmt22
echo "wmt22 en-ru gold_score_threshold=-4 finished."


echo "en-ru finished."



echo "All scripts executed successfully."
