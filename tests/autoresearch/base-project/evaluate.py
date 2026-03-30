#!/usr/bin/env python3
"""Evaluator for autoresearch integration test.

Loads result.json, computes MSE on held-out test set.
Fixed Condition — do NOT modify this file.
"""

import json
import numpy as np


def generate_test_data(n_points=50, seed=99):
    """Generate held-out test data. FIXED."""
    rng = np.random.RandomState(seed)
    x = np.linspace(-np.pi, np.pi, n_points)
    y = np.sin(x) + rng.normal(0, 0.05, n_points)
    return x, y


def main():
    with open("result.json") as f:
        result = json.load(f)

    coeffs = np.array(result["coefficients"])
    x_test, y_test = generate_test_data()

    y_pred = np.polyval(coeffs, x_test)
    mse = float(np.mean((y_test - y_pred) ** 2))

    print(f"mse={mse:.6f}")


if __name__ == "__main__":
    main()
