#!/bin/bash

# Define a list of 45 unique pairs of digits
pairs=( "3 4" "3 5" "3 6" "3 7" "3 8" "3 9"
        "4 5" "4 6" "4 7" "4 8" "4 9")

# Loop through each pair of digits
for pair in "${pairs[@]}"; do
    # Run main.py with the specified arguments and current mnist_values pair
    python3 main.py --batch_size 128 --epochs 2048 --regular_lr 0.1 --slow_lr 0.1 --mnist_values $pair
    python3 main.py --batch_size 128 --epochs 2048 --regular_lr 0.1 --slow_lr 0.01 --mnist_values $pair
    python3 main.py --batch_size 128 --epochs 2048 --regular_lr 0.1 --slow_lr 0.001 --mnist_values $pair
done
