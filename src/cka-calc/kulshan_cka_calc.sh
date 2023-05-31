#!/bin/bash

# Start a new screen session named "cka-calc"
screen -dmS cka-calc

python3 -m venv slow-feature-learning-venv
activate slow-feature-learning-venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# Activate the virtual environment
source slow-feature-learning-venv/bin/activate

# Change dir to the location of the python program
cd src/cka-calc

# Loop through each pair of digits
for ((i=0; i<=8; i++)); do
    for ((j=i+1; j<=9; j++)); do
        # Combine the digits to form a digit pair string
        digits=$(printf "%d%d" $i $j)

        # Run your Python program with the digit pair string as an argument
        python cka_calc.py --mnistDigits "$digits"
    done
done
