#!/bin/bash

python scripts/py/create_performance_table.py -lp zh-en
python scripts/py/create_performance_table.py -lp zh-en --apx
python scripts/py/create_performance_table.py -lp zh-en --dev
python scripts/py/create_performance_table.py -lp zh-en --dev --apx
python scripts/py/create_performance_table.py -lp en-de --apx
python scripts/py/create_performance_table.py -lp en-de --dev --apx
python scripts/py/create_performance_table.py -lp he-en --apx
python scripts/py/create_mbr_table.py 
