#!/bin/bash
digits=(98 27 38)
for val in ${digits[@]}; 
do
    echo $val
    python main.py $val 2000 3000 200 0.01 0.0001
    python main.py $val 2000 3000 200 0.01 0.01
done