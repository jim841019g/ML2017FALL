#!/bin/bash

python3 cut.py $1
python3 pre_for_test.py
python2 predict.py $2  
