#!/usr/bin/env python3
"""Polynomial fitting trainer for autoresearch integration test.

Fits a polynomial to sin(x) sample points using numpy least squares.
Supports pressure conditions: --epoch-limit and --time-limit.

Fixed Conditions (do NOT modify):
- Dataset: sin(x) on [-pi, pi], 100 train points, seed=42
- Termination logic: epoch limit and time limit
- Output format: result.json with coefficients and train_mse
"""

import argparse
import json
import time
import numpy as np


def generate_data(n_points, seed=42):
    """Generate sin(x) training data. FIXED — do not modify."""
    rng = np.random.RandomState(seed)
    x = np.linspace(-np.pi, np.pi, n_points)
    y = np.sin(x) + rng.normal(0, 0.05, n_points)
    return x, y


def fit_polynomial(x, y, degree=2):
    """Fit polynomial of given degree. This is the Variable Condition."""
    coeffs = np.polyfit(x, y, degree)
    return coeffs


def compute_mse(x, y, coeffs):
    """Compute mean squared error."""
    y_pred = np.polyval(coeffs, x)
    return float(np.mean((y - y_pred) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=2,
                        help="Polynomial degree (Variable Condition)")
    parser.add_argument("--epoch-limit", type=int, default=1,
                        help="Max epochs (Pressure Condition — FIXED)")
    parser.add_argument("--time-limit", type=float, default=10.0,
                        help="Max seconds (Pressure Condition — FIXED)")
    args = parser.parse_args()

    start_time = time.time()

    # Generate training data (FIXED)
    x_train, y_train = generate_data(100, seed=42)

    # Training loop with pressure conditions (FIXED termination logic)
    best_coeffs = None
    best_mse = float("inf")

    for epoch in range(args.epoch_limit):
        elapsed = time.time() - start_time
        if elapsed >= args.time_limit:
            print(f"Time limit reached ({args.time_limit}s)")
            break

        coeffs = fit_polynomial(x_train, y_train, degree=args.degree)
        mse = compute_mse(x_train, y_train, coeffs)

        if mse < best_mse:
            best_mse = mse
            best_coeffs = coeffs

        print(f"Epoch {epoch+1}: degree={args.degree}, train_mse={mse:.6f}")

    # Save result (FIXED output format)
    result = {
        "coefficients": best_coeffs.tolist(),
        "degree": args.degree,
        "train_mse": best_mse,
    }
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Training complete. MSE={best_mse:.6f}")


if __name__ == "__main__":
    main()
