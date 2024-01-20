#!/bin/bash

python3 ablation.py
python3 ../plot/ablation-1b.py ../script/ablation.csv
mv ../plot/ablation-1b.pdf .
