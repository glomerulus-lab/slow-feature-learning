#!/bin/bash

# Define a list of 45 unique pairs of digits
pairs=( "5 6" "5 7" "5 8" "5 9"
        "6 7" "6 8" "6 9"
        "7 8" "7 9"
        "8 9" )

# Loop through each pair of digits
for pair in "${pairs[@]}"; do
    # Run main.py with the specified arguments and current mnist_values pair
    python3 main.py --batch_size 128 --epochs 2048 --regular_lr 0.1 --slow_lr 0.1 --mnist_values $pair
    python3 main.py --batch_size 128 --epochs 2048 --regular_lr 0.1 --slow_lr 0.01 --mnist_values $pair
    python3 main.py --batch_size 128 --epochs 2048 --regular_lr 0.1 --slow_lr 0.001 --mnist_values $pair
done
