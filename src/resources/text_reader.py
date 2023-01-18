"""
TXT File Reader Functions
"""

def read_hyperparams(filepath):
    hyper_params = {"mnist_values": [],
                    "ml_width": None,
                    "learning_rate": None,
                    "slow_learning_rate": None,
                    "batch_size": None,
                    "epochs": None}

    with open(filepath, 'r') as file:
        for line in file:
            try:
                key, value = line.strip().split()
                hyper_params[key] = float(value)
            except:
                key, value1, value2 = line.strip().split()
                hyper_params[key] = [int(value1), int(value2)]
                continue

    hyper_params['ml_width'] = int(hyper_params['ml_width'])
    hyper_params['batch_size'] = int(hyper_params['batch_size'])
    hyper_params['epochs'] = int(hyper_params['epochs'])


    return hyper_params
