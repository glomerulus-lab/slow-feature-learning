#!/bin/bash

python3 main.py --batch_size 128 --epochs 2048 --regular_lr 0.1 --slow_lr 0.01 --mnist_values 0 1

python3 main.py --batch_size 128 --epochs 2048 --regular_lr 0.1 --slow_lr 0.001 --mnist_values 0 1
