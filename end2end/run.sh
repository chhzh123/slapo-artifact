#!/bin/bash

bash run_all_single_node.sh configs/single_node_asplos24.cfg
python3 plot/single_node_1b.py single_node_v100.csv
mv single_node_v100_1b.pdf configs/
