#!/bin/bash

# Start a new screen session named "cka-calc"
screen -dmS cka-calc

# Activate the virtual environment
source ../../slow-feature-learning-venv/bin/activate

# Loop through each pair of digits
for ((i=0; i<=8; i++)); do
    for ((j=i+1; j<=9; j++)); do
        # Combine the digits to form a digit pair string
        digits=$(printf "%d%d" $i $j)

        # Run your Python program with the digit pair string as an argument
        python cka_calc.py --mnistDigits "$digits"
    done
done
