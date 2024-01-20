#!/bin/bash

cp ablation.py ../
cd ../
python3 ablation.py
python3 plot/ablation-1b.py ablation.csv
mv ablation-1b.pdf script/
