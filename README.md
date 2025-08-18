# mt-metrics-thresholds

## Setup

```bash
pip install -e .
```

## Meta-Evaluation

To run the interpretable MT meta-evaluation as in the paper in a given WMT test set and language pair, follow the steps below for the two tasks: **Data Filtering** and **Translation Re-Ranking**.

### Data Filtering

For example, to run it on WMT23, Chinese-English, you can use the command:

```bash
python scripts/py/rank_metrics.py \
    --testset-names wmt23 \
    --lps zh-en \
    --refs-to-use refA # standard reference used for zh-en in wmt23
    --task data-filtering \
    --average-by sys \
    --include-human \
    --include-outliers \
    --gold-name mqm \
    --gold-score-threshold -1 \ # -1 for PERFECT vs OTHER and -4 for GOOD vs BAD
```

To run the above command in multiprocessing mode, you can use the `--n-processes` argument, and set it to the number of processes you want to run in parallel. If it is left unset, the number of processes will be equal to the number of processors on your device.
To start from pre-computed metrics thresholds, and only run the F1-scores computation (no optimization), use the `--thresholds-from-json` argument.

### Translation Re-Ranking

For example, to run it on WMT23, Chinese-English, you can use the following command:

```bash
python scripts/py/rank_metrics.py \
    --testset-names wmt23 \
    --lps zh-en \
    --refs-to-use refA # standard reference used for zh-en in wmt23
    --task translation-reranking \
    --include-human \
    --include-outliers \
    --gold-name mqm \
```

Translation Re-Ranking is very fast and does not require an optimization process (contrary to Data Filtering), so it always runs in a single process.

