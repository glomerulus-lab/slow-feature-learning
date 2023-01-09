"""
DESCR: PLOTTING PROGRAM FOR ACCURACY V.S. EPOCHS (BAR PLOT)
DATE: 01/09/2022
BY: CAMERON KAMINSKI
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from resources.json_fn import merge_on_metric

nrow = 2
ncol = 2
fig, axes = plt.subplots(2, 2)
title = ["[0 1]",  "[2 7]", "[4 6]", "[8 9]"]

# Loading the data.
cka_dfs = [None]*4
cka_dfs[0] = merge_on_metric('01', "Centered Kernel Alignment")
cka_dfs[1] = merge_on_metric('27', "Centered Kernel Alignment")
cka_dfs[2] = merge_on_metric('46', "Centered Kernel Alignment")
cka_dfs[3] = merge_on_metric('89', "Centered Kernel Alignment")

for i, ax in enumerate(axes.ravel()):

    # Generate 16 unique colors
    colors = sns.color_palette("hls", 16)
    ax.set_title(f"Digits -> {title[i]}")
    ax.set_ylabel('CKA')
    ax.set_xlabel('Learning Rates')
    ax.set_ylim([0.4, 1])
    cka_dfs[i].iloc[-1].plot.bar(ax=ax)

plt.show()