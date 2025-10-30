- Optimized Global Minimum Variance portfolios comparing sample covariance vs. Ledoit-Wolf shrinkage methods across rolling backtesting periods (4-60 months), with interactive stock selection and user-defined weight constraints validated through coverage checks for data quality.

- Implemented quadratic optimization using CVXPY (OSQP solver) with enforced covariance matrix symmetry, demonstrating Ledoit-Wolf's superior out-of-sample performance through analysis of cumulative returns, portfolio variance, weight stability, and monthly turnover metrics.

- Tech Stack: pandas, numpy, matplotlib, wrds, scipy, seaborn, cvxpy
