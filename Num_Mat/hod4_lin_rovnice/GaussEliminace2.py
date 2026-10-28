"""
GaussEliminace2.py
==================
Solves a system of linear equations A * x = b using Gaussian elimination
with partial pivoting (row swapping) followed by back substitution.

Partial Pivoting (Částečný výběr pivota):
----------------------------------------
In standard Gaussian elimination, division by small or zero diagonal elements
can cause numerical instability or division by zero.
Partial pivoting mitigates this by searching for the row with the largest
absolute value in the current column (the pivot column) at or below the diagonal:
    p = argmax_{k <= i < n} |A[i][k]|
If p != k, rows k and p in both matrix A and vector b are swapped before
performing elimination for step k. This guarantees |A[i][k] / A[k][k]| <= 1,
minimizing the growth of rounding errors.

Input File Format ('inpGaussEliminace.dat'):
--------------------------------------------
- Line 1:       n (order of square matrix A)
- Lines 2..n+1: Matrix entries column by column (each line contains the n elements
                of one column: line 2 = column 1, line 3 = column 2, etc.)
- Line n+2:     Vector b (the n elements of the right-hand side)

Scientific Notation Support:
----------------------------
Supports standard scientific notation (e.g., 1.5e-3, 2.0E+2) as well as
Fortran-style double precision notation (e.g., 5.0d0, 1.23D-4, 4.0d0).

How to Run:
-----------
1. Default run (reads 'inpGaussEliminace.dat' from the script or current directory):
       python GaussEliminace2.py

2. Custom input file:
       python GaussEliminace2.py path/to/your_input_file.dat
"""

import sys
import os
import re
from typing import List, Tuple


def parse_number(token: str) -> float:
    """
    Parses a string into a float.
    Supports standard floats (1.0, -2.5e-3) and Fortran scientific notation (5.0d0, 1.2D+3).
    """
    token = token.strip().rstrip(",")
    # Replace Fortran exponent characters 'd' or 'D' with standard 'e'
    token_clean = token.replace("d", "e").replace("D", "e")
    return float(token_clean)


def read_input_file(filepath: str) -> Tuple[int, List[List[float]], List[float]]:
    """
    Reads matrix dimension n, matrix A, and vector b from the specified file.
    
    Structure:
      - Line 1: n (size of system)
      - Next n lines: columns 0 to n-1 of matrix A
      - Final line: vector b
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # Strip comments (starting with # or !) and trailing whitespace
            clean = re.sub(r"[#!].*$", "", line).strip()
            if clean:
                lines.append(clean)

    if not lines:
        raise ValueError(f"File '{filepath}' is empty or contains only comments.")

    # 1. Read dimension n
    n = int(lines[0].split()[0])
    if n <= 0:
        raise ValueError(f"Matrix dimension n must be positive, got {n}.")

    if len(lines) < n + 2:
        raise ValueError(
            f"Expected at least {n + 2} data lines (1 for n, {n} for columns, 1 for b), "
            f"but found {len(lines)} lines."
        )

    # 2. Read matrix A column by column
    # A[i][j] represents row i, column j
    A = [[0.0] * n for _ in range(n)]
    for j in range(n):
        tokens = lines[j + 1].replace(",", " ").split()
        if len(tokens) != n:
            raise ValueError(
                f"Column {j + 1} (line {j + 2}) contains {len(tokens)} entries; expected {n}."
            )
        for i in range(n):
            A[i][j] = parse_number(tokens[i])

    # 3. Read vector b
    b_tokens = lines[n + 1].replace(",", " ").split()
    if len(b_tokens) != n:
        raise ValueError(
            f"Vector b (line {n + 2}) contains {len(b_tokens)} entries; expected {n}."
        )
    b = [parse_number(tok) for tok in b_tokens]

    return n, A, b


def gauss_elimination_partial_pivot(
    n: int,
    A_in: List[List[float]],
    b_in: List[float],
    verbose: bool = True,
    tol: float = 1e-15
) -> List[float]:
    """
    Solves Ax = b using explicit Gaussian elimination with partial pivoting
    and back substitution.
    
    Parameters:
      n       : order of matrix A
      A_in    : n x n coefficient matrix (list of lists)
      b_in    : right-hand side vector of length n (list)
      verbose : whether to print step-by-step pivoting and elimination details
      tol     : numerical zero tolerance for singularity checks
      
    Returns:
      x       : solution vector of length n
    """
    # Create working copies so original input matrix and vector are preserved
    A = [row[:] for row in A_in]
    b = list(b_in)

    # =========================================================================
    # 1. Forward Elimination with Partial Pivoting (Přímý chod s pivotací)
    # =========================================================================
    # Process each pivot column k from 0 to n-2
    for k in range(n - 1):
        # Step 1.1: Partial pivoting - find pivot row with maximum absolute entry
        # in column k at or below row k
        pivot_row = k
        max_val = abs(A[k][k])

        for i in range(k + 1, n):
            current_abs = abs(A[i][k])
            if current_abs > max_val:
                max_val = current_abs
                pivot_row = i

        # Check for singular or near-singular matrix
        if max_val < tol:
            raise ZeroDivisionError(
                f"Matrix is singular or near-singular: maximum pivot in column {k + 1} "
                f"is {max_val:.2e} (< {tol}). Cannot continue."
            )

        # Step 1.2: Row swap if the maximum pivot is not on the diagonal
        if pivot_row != k:
            if verbose:
                print(f"  [Step {k + 1}] Swapping row {k + 1} with row {pivot_row + 1} "
                      f"(pivot: A[{pivot_row + 1}][{k + 1}] = {A[pivot_row][k]:.6f})")
            # Swap rows in matrix A
            A[k], A[pivot_row] = A[pivot_row], A[k]
            # Swap corresponding entries in right-hand side vector b
            b[k], b[pivot_row] = b[pivot_row], b[k]
        else:
            if verbose:
                print(f"  [Step {k + 1}] Pivot already on diagonal: A[{k + 1}][{k + 1}] = {A[k][k]:.6f}")

        # Step 1.3: Eliminate subdiagonal entries in column k for rows k+1 to n-1
        pivot = A[k][k]
        for i in range(k + 1, n):
            factor = A[i][k] / pivot

            # Update row i of matrix A for elements from k+1 to n-1
            for j in range(k + 1, n):
                A[i][j] -= factor * A[k][j]

            # Explicitly set the eliminated entry to 0.0
            A[i][k] = 0.0

            # Update row i of vector b
            b[i] -= factor * b[k]

    # Verify that the last diagonal element is non-zero
    if abs(A[n - 1][n - 1]) < tol:
        raise ZeroDivisionError(
            f"Matrix is singular: last diagonal element A[{n}][{n}] = {A[n - 1][n - 1]:.2e} (< {tol})."
        )

    # =========================================================================
    # 2. Back Substitution (Zpětný chod)
    # =========================================================================
    # Initialize solution vector x with zeros
    x = [0.0] * n

    # Compute the last component: x[n-1] = b[n-1] / A[n-1][n-1]
    x[n - 1] = b[n - 1] / A[n - 1][n - 1]

    # Compute the remaining components backwards: from row n-2 down to 0
    for i in range(n - 2, -1, -1):
        tsum = 0.0
        for j in range(i + 1, n):
            tsum += A[i][j] * x[j]
        x[i] = (b[i] - tsum) / A[i][i]

    return x


def main():
    # Determine input file path: command line argument or default 'inpGaussEliminace.dat'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Check current working directory first, then script's directory
        candidate_cwd = "inpGaussEliminace.dat"
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
        candidate_script = os.path.join(script_dir, "inpGaussEliminace.dat")

        if os.path.isfile(candidate_cwd):
            filepath = candidate_cwd
        elif os.path.isfile(candidate_script):
            filepath = candidate_script
        else:
            filepath = candidate_cwd

    print("=" * 65)
    print(" Gaussian Elimination with Partial Pivoting (GaussEliminace2)")
    print("=" * 65)
    print(f"Reading input from: {filepath}\n")

    # Read the system from file
    try:
        n, A, b = read_input_file(filepath)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    print(f"Matrix dimension n = {n}")
    print("\nOriginal coefficient matrix A:")
    for row in A:
        print("  [" + ", ".join(f"{val:12.6f}" for val in row) + "]")

    print("\nRight-hand side vector b:")
    print("  [" + ", ".join(f"{val:12.6f}" for val in b) + "]")

    # Solve the system with partial pivoting
    print("\nElimination and Pivoting Progress:")
    try:
        x = gauss_elimination_partial_pivot(n, A, b, verbose=True)
    except ZeroDivisionError as err:
        print(f"\nComputation failed: {err}")
        sys.exit(1)

    # Print the resulting solution vector x
    print("\n" + "-" * 65)
    print(" x:")
    for i in range(n):
        print(f"  x({i + 1}) = {x[i]: .10f}")
    print("-" * 65)

    # Verification: compute residual vector r = A*x - b and max norm
    residual = []
    for i in range(n):
        ax_i = sum(A[i][j] * x[j] for j in range(n))
        residual.append(ax_i - b[i])

    max_residual = max(abs(r) for r in residual)
    print(f"\nVerification max |(Ax - b)_i| = {max_residual:.3e}")
    if max_residual < 1e-10:
        print("Status: Solution is accurate.")
    else:
        print("Status: Caution, high residual.")


if __name__ == "__main__":
    main()
