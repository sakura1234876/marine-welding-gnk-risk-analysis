#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw a reference-style Fig. 14 for Kendall correlation coefficients.

The reference paper visualizes Kendall correlation coefficients as 3D line
curves, where both horizontal axes are the parameter values and the vertical
axis is the correlation coefficient.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


OUTPUT_DIR = Path("weight_parameter_figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    xlsx_files = glob.glob("*.xlsx")
    if not xlsx_files:
        raise FileNotFoundError("No .xlsx file found in current folder.")

    xlsx = xlsx_files[0]
    xl = pd.ExcelFile(xlsx)
    kendall_raw = pd.read_excel(xlsx, sheet_name=xl.sheet_names[2])

    labels = list(kendall_raw.iloc[:, 0].astype(str))
    alphas = np.array([float(label.split("=")[1]) for label in labels])
    kendall_matrix = kendall_raw.iloc[:, 1:].to_numpy(dtype=float)

    # To keep the figure readable, draw curves every 0.05, similar to the
    # reference article's coarse parameter display.
    selected_indices = [i for i, a in enumerate(alphas) if abs((a * 100) % 5) < 1e-9]
    if np.where(np.isclose(alphas, 0.50))[0][0] not in selected_indices:
        selected_indices.append(int(np.where(np.isclose(alphas, 0.50))[0][0]))
    selected_indices = sorted(set(selected_indices))

    fig = plt.figure(figsize=(7.2, 5.4), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab20(np.linspace(0, 1, len(selected_indices)))
    for color, row_idx in zip(colors, selected_indices):
        x = alphas
        y = np.full_like(alphas, alphas[row_idx])
        z = kendall_matrix[row_idx, :]
        linewidth = 2.4 if np.isclose(alphas[row_idx], 0.50) else 1.7
        ax.plot(x, y, z, color=color, linewidth=linewidth)

    # Mark the selected alpha=0.50 curve with a black dashed projection line.
    idx_05 = int(np.where(np.isclose(alphas, 0.50))[0][0])
    ax.plot(
        alphas,
        np.full_like(alphas, 0.50),
        kendall_matrix[idx_05, :],
        color="black",
        linewidth=2.8,
        linestyle="--",
        label=r"Selected $\alpha=0.50$",
    )

    ax.set_xlabel(r"$\alpha_i$", labelpad=8)
    ax.set_ylabel(r"$\alpha_j$", labelpad=8)
    ax.set_zlabel("Correlation coefficient", labelpad=8)
    ax.set_title("Results of the Kendall Correlation Coefficient", pad=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-1, 1.05)
    ticks = np.arange(0, 1.01, 0.2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(np.arange(-1.0, 1.01, 0.5))

    # Camera angle close to the visual style of the reference figure.
    ax.view_init(elev=22, azim=-62)
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False)

    output = OUTPUT_DIR / "Fig14_reference_style_Kendall_3D.png"
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
