#!/bin/bash

# Define a list of 45 unique pairs of digits
pairs=( "0 2" "0 3" "0 4" "0 5" "0 6" "0 7" "0 8" "0 9"
        "1 2" "1 3" "1 4" "1 5" "1 6" "1 7" "1 8" "1 9"
        "2 3" "2 4" "2 5" "2 6" "2 7" "2 8" "2 9")

# Loop through each pair of digits
for pair in "${pairs[@]}"; do
    # Run main.py with the specified arguments and current mnist_values pair
    python3 main.py --batch_size 128 --epochs 2048 --regular_lr 0.1 --slow_lr 0.1 --mnist_values $pair
    python3 main.py --batch_size 128 --epochs 2048 --regular_lr 0.1 --slow_lr 0.01 --mnist_values $pair
    python3 main.py --batch_size 128 --epochs 2048 --regular_lr 0.1 --slow_lr 0.001 --mnist_values $pair

done
