#!/bin/bash

# Define an array of digits from 0 to 9
digits=(0 1 2 3 4 5 6 7 8 9)

# Loop through each pair of digits
for i in "${!digits[@]}"; do
  for j in "${!digits[@]}"; do
    if [ $i -lt $j ]; then
      # Construct the mnist_values argument with the current pair of digits
      mnist_values="${digits[i]} ${digits[j]}"

      # Run main.py with the specified arguments and current mnist_values pair
      python3 main.py --batch_size 64 --epochs 1 --regular_lr 0.1 --slow_lr 0.1 --mnist_values $mnist_values
    fi
  done
done