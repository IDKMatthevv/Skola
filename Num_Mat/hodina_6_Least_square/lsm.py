"""
lsm.py – Weighted Least-Squares Approximation (3rd-degree polynomial)
======================================================================
Fits experimental data to f(x) = a + b*x + c*x^2 + d*x^3 using the
weighted least-squares method.

Input  : data2.txt  –  three whitespace-separated columns (x_i, y_i, w_i)
Output : polynomial coefficients, fit quality, and a plot.

Usage:
    python lsm.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1.  Read the data
# ---------------------------------------------------------------------------
# Each row: x_i   y_i   w_i   (whitespace-separated)
data = np.loadtxt("data2.txt")       # shape (n, 3)
x = data[:, 0]                       # independent variable
y = data[:, 1]                       # dependent variable
w = data[:, 2]                       # weights  (0 <= w_i <= 1)

n_total    = len(x)                  # total number of data points
n_positive = int(np.sum(w > 0))      # points that actually contribute

# ---------------------------------------------------------------------------
# 2.  Build the weighted normal equations
# ---------------------------------------------------------------------------
# Polynomial degree
degree = 3

# Vandermonde-like design matrix:  Phi[i, j] = x_i^j ,  j = 0, 1, 2, 3
Phi = np.column_stack([x**j for j in range(degree + 1)])   # shape (n, 4)

# Weight matrix  W = diag(w)
W = np.diag(w)

# Normal equations:  (Phi^T W Phi) c = Phi^T W y
#   where c = [a, b, c, d]^T
A = Phi.T @ W @ Phi          # 4×4 matrix
b_vec = Phi.T @ W @ y        # 4×1 right-hand side

# Solve for the coefficients
coeffs = np.linalg.solve(A, b_vec)   # [a, b, c, d]
a, b, c, d = coeffs

# ---------------------------------------------------------------------------
# 3.  Compute the weighted sum of squared residuals
# ---------------------------------------------------------------------------
residuals = y - Phi @ coeffs
S = float(residuals.T @ W @ residuals)   # S = Σ w_i (y_i − f(x_i))^2

# ---------------------------------------------------------------------------
# 4.  Print the results
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Weighted Least-Squares Fit  –  f(x) = a + bx + cx² + dx³")
print("=" * 60)
print(f"  Total data points          : {n_total}")
print(f"  Points with positive weight: {n_positive}")
print("-" * 60)
print(f"  a = {a: .6f}")
print(f"  b = {b: .6f}")
print(f"  c = {c: .6f}")
print(f"  d = {d: .6f}")
print("-" * 60)
print(f"  f(x) = {a:.6f} + ({b:.6f})*x + ({c:.6f})*x^2 + ({d:.6f})*x^3")
print("-" * 60)
print(f"  Weighted sum of squared residuals  S = {S:.6f}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 5.  Plot the data and the fitted curve
# ---------------------------------------------------------------------------
# Dense grid for a smooth curve
x_smooth = np.linspace(x.min(), x.max(), 500)
y_smooth = a + b * x_smooth + c * x_smooth**2 + d * x_smooth**3

plt.figure(figsize=(9, 5))

# Scatter plot of original data (marker size proportional to weight)
sizes = 30 + 120 * w        # make zero-weight points small but visible
plt.scatter(x, y, s=sizes, c="royalblue", edgecolors="black",
            zorder=5, label="Experimental data")

# Fitted polynomial curve
plt.plot(x_smooth, y_smooth, color="crimson", linewidth=2,
         label=f"Fit: {a:.3f} + {b:.3f}x + {c:.3f}x² + {d:.3f}x³")

plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.title("Weighted Least-Squares Fit (3rd-degree polynomial)", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("lsm_fit.png", dpi=150)   # save a copy next to the script
plt.show()
