#!/bin/sh
digits=(01 27 38)
for val in ${digits[@]}; 
do
    echo $val
    python main.py 01 2000 3000 0.01 0.001
done