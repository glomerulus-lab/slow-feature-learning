#!/bin/bash
digits=(01 27 89 46)
for val in ${digits[@]}; 
do
    python main.py $val 2000 3000 200 0.1 0.01
    python main.py $val 2000 3000 200 0.1 0.001
    python main.py $val 2000 3000 200 0.1 0.1
    python main.py $val 2000 3000 200 0.1 0.9

    python main.py $val 2000 3000 200 0.01 0.01
    python main.py $val 2000 3000 200 0.01 0.001
    python main.py $val 2000 3000 200 0.01 0.1
    python main.py $val 2000 3000 200 0.01 0.9

    python main.py $val 2000 3000 200 0.001 0.01
    python main.py $val 2000 3000 200 0.001 0.001
    python main.py $val 2000 3000 200 0.001 0.1
    python main.py $val 2000 3000 200 0.001 0.9

    python main.py $val 2000 3000 200 0.9 0.01
    python main.py $val 2000 3000 200 0.9 0.001
    python main.py $val 2000 3000 200 0.9 0.1
    python main.py $val 2000 3000 200 0.9 0.9
done
