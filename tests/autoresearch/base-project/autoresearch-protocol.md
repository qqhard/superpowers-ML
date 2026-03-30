# Autoresearch Protocol: Polynomial Sin(x) Fitting

## Research Question
Find the optimal polynomial fitting strategy to approximate y = sin(x) on [-π, π].

## Environment
- Base code: . (current directory)
- Dataset: sin(x) on [-π, π], 100 train / 50 test points, numpy generated
- Framework: numpy only

## Fixed Conditions
- Dataset generation: seed=42 for train, seed=99 for test, point counts fixed
- Termination logic: epoch_limit and time_limit in train.py (do NOT modify)
- Evaluation logic: evaluate.py (do NOT modify)
- Output format: result.json with coefficients, degree, train_mse

## Pressure Conditions
- time_limit: 10s
- epoch_limit: 1
- Whichever triggers first

## Variable Conditions
- Polynomial degree: any integer >= 1
- Feature engineering: any (e.g., Chebyshev basis, trigonometric features)
- Regularization: any (e.g., ridge via manual implementation)
- Any modification to fit_polynomial() function in train.py

## Evaluation
- metric: mse
- direction: minimize
- eval_command: "python evaluate.py"
- baseline: 0.223000

## Termination
- max_rounds: 5
- target: 0.01

## Agent Boundary
- agent_a: ["design", "code", "train"]
- agent_b: ["evaluate", "review", "record"]
