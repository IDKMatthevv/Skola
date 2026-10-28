"""
GaussEliminace1.py
==================
Solves a system of linear equations A * x = b using Gaussian elimination
followed by back substitution.

Input File Format ('inpGaussEliminace.dat'):
--------------------------------------------
- Line 1:       n (order of square matrix A)
- Lines 2..n+1: Matrix entries column by column (each line contains the n elements
                of one column: line 2 = column 1, line 3 = column 2, etc.)
- Line n+2:     Vector b (the n elements of the right-hand side)

Scientific Notation Support:
----------------------------
Supports standard scientific notation (e.g., 1.5e-3, 2.0E+2) as well as
Fortran-style double precision notation (e.g., 5.0d0, 1.23D-4).

How to Run:
-----------
1. Default run (reads 'inpGaussEliminace.dat' from script or current directory):
       python GaussEliminace1.py

2. Custom input file:
       python GaussEliminace1.py path/to/your_input_file.dat
"""

import sys
import os
import re


def parse_number(token: str) -> float:
    """
    Parses a string into a float.
    Supports standard floats (1.0, -2.5e-3) and Fortran scientific notation (5.0d0, 1.2D+3).
    """
    token = token.strip().rstrip(",")
    # Replace Fortran exponent characters 'd' or 'D' with 'e'
    token_clean = token.replace("d", "e").replace("D", "e")
    return float(token_clean)


def read_input_file(filepath: str):
    """
    Reads order n, matrix A, and vector b from the specified file.
    
    Structure:
      - 1st line: n
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

    # 1. Read order n
    n = int(lines[0].split()[0])
    if n <= 0:
        raise ValueError(f"Matrix dimension n must be positive, got {n}.")

    if len(lines) < n + 2:
        raise ValueError(
            f"Expected at least {n + 2} data lines (1 for n, {n} for columns, 1 for b), "
            f"but found {len(lines)} lines."
        )

    # 2. Read matrix A column by column
    # A[i][j] is row i, column j
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


def gauss_elimination(n: int, A_in: list, b_in: list):
    """
    Solves Ax = b using explicit Gaussian elimination (forward elimination)
    and back substitution.
    
    Parameters:
      n    : order of matrix A
      A_in : n x n matrix (list of lists)
      b_in : vector of length n (list)
      
    Returns:
      x    : solution vector of length n
    """
    # Create working copies so original data is not modified
    A = [row[:] for row in A_in]
    b = list(b_in)

    # =========================================================================
    # 1. Forward Elimination (Přímý chod)
    # =========================================================================
    # Eliminates entries below diagonal column by column (k = 0 to n-2)
    for k in range(n - 1):
        pivot = A[k][k]
        if abs(pivot) < 1e-15:
            raise ZeroDivisionError(
                f"Zero or near-zero pivot encountered at A[{k}][{k}] = {pivot}. "
                "Gaussian elimination without pivoting cannot proceed."
            )

        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]

            # Update row i of matrix A for elements from k+1 to n-1
            for j in range(k + 1, n):
                A[i][j] = A[i][j] - factor * A[k][j]

            # Explicitly zero out the eliminated subdiagonal entry
            A[i][k] = 0.0

            # Update row i of right-hand side vector b
            b[i] = b[i] - factor * b[k]

    # Check last diagonal element
    if abs(A[n - 1][n - 1]) < 1e-15:
        raise ZeroDivisionError(
            f"Zero or near-zero diagonal element at A[{n - 1}][{n - 1}] = {A[n - 1][n - 1]}."
        )

    # =========================================================================
    # 2. Back Substitution (Zpětný chod)
    # =========================================================================
    # Computes unknown vector x starting from the last component x[n-1] down to x[0]
    x = [0.0] * n

    # Last component
    x[n - 1] = b[n - 1] / A[n - 1][n - 1]

    # Remaining components backwards
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

    print("=" * 60)
    print(" Gaussian Elimination with Back Substitution (GaussEliminace1)")
    print("=" * 60)
    print(f"Reading input from: {filepath}\n")

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

    # Solve the system
    try:
        x = gauss_elimination(n, A, b)
    except ZeroDivisionError as err:
        print(f"\nComputation failed: {err}")
        sys.exit(1)

    # Print solution x
    print("\n" + "-" * 60)
    print(" x:")
    for i in range(n):
        print(f"  x({i + 1}) = {x[i]: .10f}")
    print("-" * 60)

    # Verification: compute residual r = A*x - b
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
