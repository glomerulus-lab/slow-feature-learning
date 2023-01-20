#!/bin/bash
digits=(01 27 89 46)
for val in ${digits[@]}; 
do
    python main.py $val 2000 3000 200 0.1 0.1
    python main.py $val 2000 3000 200 0.1 0.075
    python main.py $val 2000 3000 200 0.1 0.05
    python main.py $val 2000 3000 200 0.1 0.025
    python main.py $val 2000 3000 200 0.1 0.01
    python main.py $val 2000 3000 200 0.1 0.0075
    python main.py $val 2000 3000 200 0.1 0.005
    python main.py $val 2000 3000 200 0.1 0.0025
    python main.py $val 2000 3000 200 0.1 0.001
    python main.py $val 2000 3000 200 0.1 0.00075
    python main.py $val 2000 3000 200 0.1 0.0005
    python main.py $val 2000 3000 200 0.1 0.00025
    python main.py $val 2000 3000 200 0.1 0.0001
done
