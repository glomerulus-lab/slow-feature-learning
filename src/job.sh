#!/bin/bash
digits=(01 27 89)
for val in ${digits[@]}; 
do
    python main.py $val 2000 3000 200 0.01 0.001
    python main.py $val 2000 3000 200 0.01 0.01
done
