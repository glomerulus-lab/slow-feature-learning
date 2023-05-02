#
# BENCHMARK TESTS
# BY : CAMERON KAMINSKI
# DATE : 05.01.2023
#

import torch
import time
import csv
import argparse


# Parse command line argument for file name.
parser = argparse.ArgumentParser(description='Benchmark tests for CKA')
parser.add_argument('--fileName', type=str, default='benchmark.csv', 
                    help='Name of file to write benchmark results to.')
arg = parser.parse_args()

# Function decorator to measure execution time
def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Elapsed time for {func.__name__}: {elapsed_time}')
        return elapsed_time
    return wrapper


def kernel_centering(K, device='cpu'):
    row_means = K.mean(dim=1, keepdim=True)
    col_means = K.mean(dim=0, keepdim=True)
    total_mean = K.mean()

    return (K - row_means - col_means + total_mean).to(device)


def vector_centering(v, device='cpu'):
    mean = torch.mean(v)
    centered_v = v - mean
    return centered_v.to(device)


@time_decorator
def cka_new(y, phi, device='cpu'):
    yc = vector_centering(y, device=device)
    K1c = yc.T @ yc
    phic = kernel_centering(phi, device=device)
    v = phic.T @ yc
    return (v.T @ v)/(torch.norm(y @ y.T)*torch.norm(phi @ phi.T)).to(device)


@time_decorator
def cka_est(y, phi, device='cpu'):
    y = vector_centering(y, device=device)
    phic = kernel_centering(phi, device=device)
    v = phi.T @ y
    return (v.T @ v) / (y.T @ y * torch.norm(phic.T @ phic)).to(device)


@time_decorator
def cka_old(y, phi, device='cpu'):
    yc = vector_centering(y.T, device=device)
    K1c = yc.T @ yc
    phic = kernel_centering(phi, device=device)
    K2c = phic @ phic.T
    return torch.trace(K1c @ K2c)/(torch.norm(K1c)*torch.norm(K2c)).to(device)



benchmarks = {
    'cka_new': [],
    'cka_est': [],
    'cka_old': []
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Run each function and store the timing and result in the benchmarks dictionary
for i in range(16):
    y = torch.randn(int(12000 / ((i + 1) / 16)),
                    1).double()
    phi = torch.randn(int(12000 / ((i + 1) / 16)), 
                          int(1000 / ((i + 1) / 16))).double()
    benchmarks['cka_new'].append(time_decorator(cka_new)(y, phi))
    benchmarks['cka_est'].append(time_decorator(cka_est)(y, phi))
    benchmarks['cka_old'].append(time_decorator(cka_old)(y, phi))

with open('benchmarks.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(benchmarks.keys())
    writer.writerow(benchmarks.values())

