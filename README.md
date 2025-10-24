<div align="center">

# Beyond Correlation: Interpretable Evaluation of Machine Translation Metrics 

[![Conference](https://img.shields.io/badge/EMNLP-2024-4b44ce
)](https://2024.emnlp.org/)
[![Paper](http://img.shields.io/badge/paper-EMNLP--anthology-B31B1B.svg)](https://aclanthology.org/2024.emnlp-main.1152/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

</div>

This repository provides a research toolkit to run interpretable meta-evaluation of Machine Translation (MT) metrics. It offers commands for evaluating metrics capabilities across two metric tasks: 
- **Data Filtering**: Separate good-quality translations from poor-quality ones.
- **Translation Re‑Ranking**: Identify the best translation in a pool of translations of the same source text.

## Setup

```bash
pip install -e .
```

## Data Filtering

Evaluate metrics capabilities in data filtering using the `rank_metrics.py` script passing `data-filtering` as the task parameter.

For example, run the evaluation using the WMT23 test set in the Chinese to English translation direction using the following command:

```bash
python scripts/py/rank_metrics.py \
    --testset-names wmt23 \
    --lps zh-en \
    --refs-to-use refA
    --task data-filtering \
    --average-by sys \
    --include-human \
    --include-outliers \
    --gold-name mqm \
    --gold-score-threshold -1 \ # Use -1 for PERFECT vs OTHER and -4 for GOOD vs BAD
```

For each metric, this command runs an optimization process to find the best score threshold to separate between GOOD and BAD translations, or between PERFECT and OTHER translations, depending on the `gold-score-threshold` passed as parameter. To speed it up, you can pass the `--n-processes` argument to set the number of processes you want to run in parallel. If left unset, the number of processes will be equal to the number of processors on your device.

This command works also with pre-computed metrics thresholds. To do so, use the `--thresholds-from-json` argument. This way, it will only compute Precision, Recall, and F1-scores, without running the slower optimization process.

### Translation Re-Ranking

Evaluate metrics capabilities in translation re-ranking using the `rank_metrics.py` script passing `translation-reranking` as the task parameter.

For example, run the evaluation using the WMT23 test set in the Chinese to English translation direction using the following command:

```bash
python scripts/py/rank_metrics.py \
    --testset-names wmt23 \
    --lps zh-en \
    --refs-to-use refA
    --task translation-reranking \
    --include-human \
    --include-outliers \
    --gold-name mqm \
```

For each metric, this command measures its re-ranking precision as the number of times it is able to identify the best translation out of a pool of translations of the same source.

## Cite us

If you find our paper or code useful, please reference this work:

```bibtex
@inproceedings{perrella-etal-2024-beyond,
    title = "Beyond Correlation: Interpretable Evaluation of Machine Translation Metrics",
    author = "Perrella, Stefano  and
      Proietti, Lorenzo  and
      Huguet Cabot, Pere-Llu{\'i}s  and
      Barba, Edoardo  and
      Navigli, Roberto",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1152/",
    doi = "10.18653/v1/2024.emnlp-main.1152",
    pages = "20689--20714",
}
```

## License

This work is under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
