"""
GaussEliminace3.py
==================
Solves a system of linear equations A * x = b using Gaussian elimination
with complete (total) pivoting (Gaussova eliminace s úplnou pivotací)
followed by back substitution.

Theoretical Overview:
---------------------
In complete pivoting, at each step k (k = 0, ..., n-2), the algorithm searches
the entire active lower-right submatrix A[k:n, k:n] for the element with the
maximum absolute value:
    |A[p][q]| = max { |A[i][j]| : k <= i < n, k <= j < n }

If the pivot element is not at (k, k):
  1. Row swap: If p != k, row k and row p are swapped in matrix A and vector b.
     This corresponds to reordering equations.
  2. Column swap: If q != k, column k and column q are swapped across all rows in A.
     This corresponds to reordering the unknowns x, which requires tracking the
     permutation of column indices throughout the computation.

Complete pivoting provides the highest numerical stability among standard
Gaussian elimination variants by minimizing error propagation from round-off,
at the cost of O(n^3) search comparisons.

After forward elimination produces an upper-triangular matrix U, back substitution
computes the permuted unknowns x_tilde. Finally, the solution vector x is restored
to its original order using the tracked column permutations.

Input File Format ('inpGaussEliminace.dat'):
--------------------------------------------
- Line 1:       n (order of square matrix A)
- Lines 2..n+1: Matrix entries column by column (each line contains the n elements
                of one column: line 2 = column 1, line 3 = column 2, etc.)
- Line n+2:     Vector b (the n elements of the right-hand side)

Scientific Notation Support:
----------------------------
Supports standard scientific notation (e.g., 1.5e-3, 2.0E+2) as well as
Fortran-style double precision notation (e.g., 5.0d0, 1.23D-4, 4.0D+00).

How to Run:
-----------
1. Default run (reads 'inpGaussEliminace.dat' from current directory or script folder):
       python GaussEliminace3.py

2. Custom input file path:
       python GaussEliminace3.py path/to/your_input_file.dat
"""

import sys
import os
import re
from typing import List, Tuple


def parse_number(token: str) -> float:
    """
    Parses a string token into a floating-point number.
    Supports standard floats (1.0, -2.5e-3) and Fortran scientific notation (5.0d0, 1.2D+3).
    """
    token = token.strip().rstrip(",")
    # Replace Fortran exponent characters 'd' or 'D' with standard 'e'
    token_clean = token.replace("d", "e").replace("D", "e")
    return float(token_clean)


def read_input_file(filepath: str) -> Tuple[int, List[List[float]], List[float]]:
    """
    Reads matrix order n, coefficient matrix A, and vector b from a data file.

    Format:
      - Line 1:       Integer n (dimension)
      - Next n lines: Columns 0 to n-1 of matrix A (column by column)
      - Final line:   Vector b (elements b_1 to b_n)

    Ignores blank lines and comments starting with '#' or '!'.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # Strip comments (starting with # or !) and surrounding whitespace
            clean = re.sub(r"[#!].*$", "", line).strip()
            if clean:
                lines.append(clean)

    if not lines:
        raise ValueError(f"File '{filepath}' is empty or contains only comments.")

    # 1. Read matrix order n
    n = int(lines[0].split()[0])
    if n <= 0:
        raise ValueError(f"Matrix dimension n must be positive, got {n}.")

    if len(lines) < n + 2:
        raise ValueError(
            f"Expected at least {n + 2} data lines (1 for n, {n} for columns of A, 1 for b), "
            f"but found {len(lines)} lines."
        )

    # 2. Read matrix A column by column
    # In the file, line (j + 2) represents column j:
    #   tokens[i] corresponds to A[i][j] (row i, column j)
    A = [[0.0] * n for _ in range(n)]
    for j in range(n):
        tokens = lines[j + 1].replace(",", " ").split()
        if len(tokens) != n:
            raise ValueError(
                f"Column {j + 1} (line {j + 2}) contains {len(tokens)} entries; expected {n}."
            )
        for i in range(n):
            A[i][j] = parse_number(tokens[i])

    # 3. Read right-hand side vector b
    b_tokens = lines[n + 1].replace(",", " ").split()
    if len(b_tokens) != n:
        raise ValueError(
            f"Vector b (line {n + 2}) contains {len(b_tokens)} entries; expected {n}."
        )
    b = [parse_number(tok) for tok in b_tokens]

    return n, A, b


def gauss_elimination_complete_pivot(
    n: int,
    A_in: List[List[float]],
    b_in: List[float],
    verbose: bool = True,
    tol: float = 1e-15,
) -> List[float]:
    """
    Solves Ax = b using explicit Gaussian elimination with complete (total) pivoting
    and back substitution.

    Parameters:
      n       : order of matrix A
      A_in    : n x n coefficient matrix (list of lists)
      b_in    : right-hand side vector of length n (list)
      verbose : whether to display step-by-step pivoting and elimination progress
      tol     : numerical zero tolerance for singularity checks

    Returns:
      x       : solution vector of length n in original unknown order
    """
    # Create working copies so original matrix and vector are preserved
    A = [row[:] for row in A_in]
    b = list(b_in)

    # Tracking list for column permutations (order of unknowns)
    # col_order[k] indicates which original unknown x_i is currently at column k
    col_order = list(range(n))

    # =========================================================================
    # 1. Forward Elimination with Complete Pivoting (Přímý chod s úplnou pivotací)
    # =========================================================================
    for k in range(n - 1):
        # Step 1.1: Complete pivoting search
        # Find entry with maximum absolute value in submatrix A[k:n][k:n]
        pivot_row = k
        pivot_col = k
        max_abs = abs(A[k][k])

        for i in range(k, n):
            for j in range(k, n):
                val = abs(A[i][j])
                if val > max_abs:
                    max_abs = val
                    pivot_row = i
                    pivot_col = j

        # Singularity check
        if max_abs < tol:
            raise ZeroDivisionError(
                f"Matrix is singular or near-singular: maximum pivot in active submatrix at step {k + 1} "
                f"is {max_abs:.2e} (< {tol}). Cannot proceed."
            )

        # Step 1.2: Row swap (if pivot row != current row k)
        # Swaps row k and pivot_row in matrix A and vector b
        if pivot_row != k:
            if verbose:
                print(
                    f"  [Step {k + 1}] Row swap: row {k + 1} <-> row {pivot_row + 1}"
                )
            A[k], A[pivot_row] = A[pivot_row], A[k]
            b[k], b[pivot_row] = b[pivot_row], b[k]

        # Step 1.3: Column swap (if pivot column != current column k)
        # Swaps column k and pivot_col across all rows, and updates column permutation tracker
        if pivot_col != k:
            if verbose:
                orig_col_k = col_order[k] + 1
                orig_col_p = col_order[pivot_col] + 1
                print(
                    f"  [Step {k + 1}] Column swap: col {k + 1} (x{orig_col_k}) <-> col {pivot_col + 1} (x{orig_col_p})"
                )
            for r in range(n):
                A[r][k], A[r][pivot_col] = A[r][pivot_col], A[r][k]
            col_order[k], col_order[pivot_col] = col_order[pivot_col], col_order[k]

        if verbose:
            print(
                f"  [Step {k + 1}] Pivot element A[{k + 1}][{k + 1}] = {A[k][k]: .6f}"
            )

        # Step 1.4: Forward elimination of subdiagonal entries in column k
        pivot = A[k][k]
        for i in range(k + 1, n):
            factor = A[i][k] / pivot

            # Update row i of matrix A for elements from k+1 to n-1
            for j in range(k + 1, n):
                A[i][j] -= factor * A[k][j]

            # Explicitly set the eliminated entry to zero
            A[i][k] = 0.0

            # Update right-hand side vector b
            b[i] -= factor * b[k]

    # Verify that the final diagonal element is non-zero
    if abs(A[n - 1][n - 1]) < tol:
        raise ZeroDivisionError(
            f"Matrix is singular: last diagonal entry A[{n}][{n}] = {A[n - 1][n - 1]:.2e} (< {tol})."
        )

    # =========================================================================
    # 2. Back Substitution (Zpětný chod)
    # =========================================================================
    # Computes permuted unknown vector x_permuted from upper triangular system U * x_permuted = b
    x_permuted = [0.0] * n

    # Last component
    x_permuted[n - 1] = b[n - 1] / A[n - 1][n - 1]

    # Remaining components backwards from n-2 down to 0
    for i in range(n - 2, -1, -1):
        tsum = 0.0
        for j in range(i + 1, n):
            tsum += A[i][j] * x_permuted[j]
        x_permuted[i] = (b[i] - tsum) / A[i][i]

    # =========================================================================
    # 3. Unpermute Solution Vector (Návrat k původnímu pořadí neznámých)
    # =========================================================================
    # col_order[k] contains the original variable index corresponding to column k
    x = [0.0] * n
    for k in range(n):
        orig_index = col_order[k]
        x[orig_index] = x_permuted[k]

    return x


def main():
    # Determine input file path: command line argument or default 'inpGaussEliminace.dat'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Check current working directory first, then script directory
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
    print(" Gaussian Elimination with Complete Pivoting (GaussEliminace3)")
    print("=" * 65)
    print(f"Reading input from: {filepath}\n")

    # Read the linear system from file
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

    # Solve the system with complete pivoting
    print("\nElimination and Complete Pivoting Progress:")
    try:
        x = gauss_elimination_complete_pivot(n, A, b, verbose=True)
    except ZeroDivisionError as err:
        print(f"\nComputation failed: {err}")
        sys.exit(1)

    # Print the resulting solution vector x
    print("\n" + "-" * 65)
    print(" x:")
    for i in range(n):
        print(f"  x({i + 1}) = {x[i]: .10f}")
    print("-" * 65)

    # Verification: compute residual vector r = A*x - b and max norm ||r||_inf
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
