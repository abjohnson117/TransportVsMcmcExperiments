#!/usr/bin/env python3
"""
plot_sample.py

Loads the generated training dataset and plots a single sample:
  left  panel: parameter field (33x33)
  right panel: PDE solution on the full grid (33x33)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "training_dataset")

SAMPLE_IDX = 0

if __name__ == "__main__":
    parameters = np.load(os.path.join(DATA_DIR, "parameters_p.npy"))
    solutions = np.load(os.path.join(DATA_DIR, "solutions_full_p.npy"))

    param_field = parameters[SAMPLE_IDX]
    solution_field = solutions[SAMPLE_IDX]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(param_field, origin="lower", interpolation="bilinear")
    axes[0].set_title("Parameter field (33x33)")
    axes[0].grid(False)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(solution_field, origin="lower", interpolation="bilinear")
    axes[1].set_title("PDE solution (33x33)")
    axes[1].grid(False)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join(HERE, "sample_plot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
