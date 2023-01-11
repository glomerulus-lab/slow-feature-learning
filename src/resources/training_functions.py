"""
Training Functions For Model
"""

def classify_targets(targets, values):
    new_targets = targets.clone()

    # Changing targets to a classifiable number.
    for key, element in enumerate(values):
        new_targets[targets == element] = key

    labels = one_hot(new_targets.long(), len(values)).to(torch.float32)

    return labels