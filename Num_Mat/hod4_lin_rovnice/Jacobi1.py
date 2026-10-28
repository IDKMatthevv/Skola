"""
Jacobi1.py
==========
Solves a system of linear equations A * x = b using the Jacobi Iterative Method
(Jacobiho iterační metoda).

Converted from the Fortran 95 program 'JacobiIterMethod.f95'.

Mathematical Background:
------------------------
For a linear system A * x = b where A is an n x n square matrix with non-zero
diagonal entries (a_ii != 0):
Matrix A can be decomposed into:
    A = D + R
where:
    - D is the diagonal matrix: D = diag(a_11, a_22, ..., a_nn)
    - R is the remainder matrix containing off-diagonal entries (L + U)

The Jacobi iteration calculates the new approximation x^(k) from the previous
approximation x^(k-1) component by component:
    x_i^(k) = (1 / a_ii) * ( b_i - sum_{j != i} (a_ij * x_j^(k-1)) )

Stopping Criterion (Ukončovací kritérium):
------------------------------------------
The iteration stops when the change between successive approximations in the
1-norm (referred to in the Fortran code comments as "sloupcová norma" / column norm)
falls below a chosen tolerance eps:
    SN = ||x^(k) - x^(k-1)||_1 = sum_{i=1}^n |x_i^(k) - x_i^(k-1)| < eps

Convergence Condition:
----------------------
Jacobi's method is guaranteed to converge if matrix A is strictly diagonally
dominant by rows:
    |a_ii| > sum_{j != i} |a_ij|   for all i = 1, ..., n

How to Run:
-----------
1. Default run (uses the built-in system from JacobiIterMethod.f95, or
   reads 'inpJacobi.dat' if present):
       python Jacobi1.py

2. Custom input file:
       python Jacobi1.py path/to/input_file.dat
"""

import sys
import os
import re
from typing import List, Tuple, Optional


def parse_number(token: str) -> float:
    """
    Parses a string token into a float.
    Supports standard floats (e.g., 1.5, -2.0e-3) as well as Fortran-style
    double precision notation (e.g., 4.0d0, -1.0D-6).
    """
    token = token.strip().rstrip(",")
    # Replace Fortran exponent 'd' or 'D' with standard 'e'
    clean_token = token.replace("d", "e").replace("D", "e")
    return float(clean_token)


def get_default_system() -> Tuple[int, float, List[List[float]], List[float], List[float]]:
    """
    Returns the default test system hardcoded in Fortran 'JacobiIterMethod.f95':
      - Dimension n = 4
      - Tolerance eps = 1.0e-6
      - Matrix A (4x4, strictly diagonally dominant and symmetric)
      - Right-hand side b = [7.0, -10.0, 4.0, 7.0]
      - Initial guess x0 = [0.0, 0.0, 0.0, 0.0]
    Exact solution: x = [1.0, -1.0, 1.0, 1.0]
    """
    n = 4
    eps = 1.0e-6

    # Data "natvrdo" do programu :-)
    A = [
        [ 4.0, -1.0,  0.0,  2.0],
        [-1.0, 10.0,  2.0, -1.0],
        [ 0.0,  2.0,  7.0, -1.0],
        [ 2.0, -1.0, -1.0,  5.0]
    ]
    b = [7.0, -10.0, 4.0, 7.0]

    # Počáteční iterace = 0
    x0 = [0.0, 0.0, 0.0, 0.0]

    return n, eps, A, b, x0


def read_input_file(filepath: str) -> Tuple[int, List[List[float]], List[float]]:
    """
    Reads order n, matrix A, and vector b from an external data file.
    
    File Structure:
      - 1st data line: Dimension n
      - Next n lines : Matrix entries (either row-by-row or column-by-column)
      - Final line   : Right-hand side vector b
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            clean = re.sub(r"[#!].*$", "", line).strip()
            if clean:
                lines.append(clean)

    if not lines:
        raise ValueError(f"Input file '{filepath}' is empty or contains only comments.")

    # 1. Read dimension n
    n = int(lines[0].split()[0])
    if n <= 0:
        raise ValueError(f"Matrix dimension n must be positive, got {n}.")

    if len(lines) < n + 2:
        raise ValueError(
            f"Expected at least {n + 2} data lines (n, {n} matrix rows/columns, b), "
            f"but found {len(lines)} lines."
        )

    # 2. Read matrix A
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        tokens = lines[i + 1].replace(",", " ").split()
        if len(tokens) != n:
            raise ValueError(
                f"Line {i + 2} has {len(tokens)} entries; expected {n}."
            )
        for j in range(n):
            A[i][j] = parse_number(tokens[j])

    # 3. Read vector b
    b_tokens = lines[n + 1].replace(",", " ").split()
    if len(b_tokens) != n:
        raise ValueError(
            f"Vector b (line {n + 2}) has {len(b_tokens)} entries; expected {n}."
        )
    b = [parse_number(tok) for tok in b_tokens]

    return n, A, b


def check_diagonal_dominance(n: int, A: List[List[float]]) -> Tuple[bool, List[str]]:
    """
    Checks if matrix A is strictly diagonally dominant by rows:
        |A[i][i]| > sum_{j != i} |A[i][j]|
    Returns a boolean flag and detailed diagnostics for each row.
    """
    is_strictly_dominant = True
    diagnostics = []

    for i in range(n):
        diag_val = abs(A[i][i])
        off_diag_sum = sum(abs(A[i][j]) for j in range(n) if i != j)
        diff = diag_val - off_diag_sum
        if diff > 1e-14:
            status = "STRICTLY DOMINANT"
        elif abs(diff) <= 1e-14:
            status = "WEAKLY DOMINANT"
            is_strictly_dominant = False
        else:
            status = "NOT DOMINANT"
            is_strictly_dominant = False

        diag_str = (
            f"Row {i + 1}: |a_{i+1}{i+1}| = {diag_val:.4f}, "
            f"sum(|a_{i+1}j|) = {off_diag_sum:.4f} -> {status}"
        )
        diagnostics.append(diag_str)

    return is_strictly_dominant, diagnostics


def jacobi(n: int, eps: float, A: List[List[float]], b: List[float],
           x_init: Optional[List[float]] = None, kmax: int = 1000,
           verbose: bool = True) -> Tuple[List[float], int, float]:
    """
    Solves A * x = b using the Jacobi Iteration Method.
    Direct translation of subroutine Jacobi(n, eps, a, b, x) from Fortran 95.

    Parameters:
      n       : Order of the matrix
      eps     : Convergence tolerance (stopping condition: ||x^(k) - x^(k-1)||_1 < eps)
      A       : n x n matrix (list of lists of floats)
      b       : Right-hand side vector of length n (list of floats)
      x_init  : Initial iteration vector x^(0); defaults to all zeros if None
      kmax    : Maximum number of iterations allowed (default: 1000)
      verbose : If True, prints iteration number and norm at each step

    Returns:
      x       : Solution vector of length n
      k       : Total number of iterations performed
      SN      : Final norm ||x^(k) - x^(k-1)||_1
    """
    # Verify non-zero diagonal entries
    for i in range(n):
        if abs(A[i][i]) < 1e-15:
            raise ZeroDivisionError(
                f"Zero diagonal entry encountered at A[{i+1}][{i+1}] = {A[i][i]}. "
                "Jacobi iteration requires non-zero diagonal elements."
            )

    # Počáteční iterace (Initial iteration vector)
    if x_init is None:
        x = [0.0] * n
    else:
        x = list(x_init)

    # x0 = x (store previous iteration vector)
    x0 = list(x)

    # Iteration loop: do k=1,kmax
    for k in range(1, kmax + 1):
        # SN = 0.0d0 ! Akumulátor pro sloupcovou normu (1-norm: sum of |x_i - x0_i|)
        SN = 0.0

        # Compute next approximation x^(k) from previous x0^(k-1)
        for i in range(n):
            # dsum = b(i)
            dsum = b[i]

            # do j=1,n; if(i == j) cycle; dsum = dsum - a(i,j)*x0(j)
            for j in range(n):
                if i == j:
                    continue
                dsum -= A[i][j] * x0[j]

            # x(i) = dsum / a(i,i)
            x[i] = dsum / A[i][i]

            # SN = SN + dabs(x(i) - x0(i)) ! Sloupcová norma
            SN += abs(x[i] - x0[i])

        # Print iteration status
        if verbose:
            print(f" iterace = {k:4d}   sloupcova norma = {SN:.10e}")

        # if(SN < eps) exit
        if SN < eps:
            return x, k, SN

        # x0 = x (update previous iteration for the next step)
        x0 = list(x)

    # If the loop finishes without breaking, k exceeded kmax
    raise RuntimeError(
        f"Jacobi: Příliš mnoho iterací! (Exceeded maximum allowed iterations: {kmax})"
    )


def main():
    print("=" * 70)
    print(" Jacobi Iterative Method for Linear Systems (Jacobi1.py)")
    print(" Converted from Fortran 95: JacobiIterMethod.f95")
    print("=" * 70)

    # 1. Determine input source: custom file, default file, or hardcoded default
    default_input_file = "inpJacobi.dat"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Reading input system from file: '{filepath}'\n")
        n, A, b = read_input_file(filepath)
        eps = 1.0e-6
        x0 = [0.0] * n
    elif os.path.isfile(default_input_file):
        print(f"Found default input file: '{default_input_file}'\n")
        n, A, b = read_input_file(default_input_file)
        eps = 1.0e-6
        x0 = [0.0] * n
    else:
        print("Using built-in default problem from JacobiIterMethod.f95\n")
        n, eps, A, b, x0 = get_default_system()

    # 2. Display system parameters
    print(f"Order of matrix n = {n}")
    print(f"Convergence tolerance eps = {eps:.1e}")

    print("\nMatrix A:")
    for i in range(n):
        row_str = ", ".join(f"{A[i][j]:10.4f}" for j in range(n))
        print(f"  Row {i + 1}: [ {row_str} ]")

    print("\nRight-hand side vector b:")
    b_str = ", ".join(f"{val:10.4f}" for val in b)
    print(f"  b = [ {b_str} ]")

    print("\nInitial guess x^(0):")
    x0_str = ", ".join(f"{val:10.4f}" for val in x0)
    print(f"  x^(0) = [ {x0_str} ]")

    # 3. Check diagonal dominance (sufficient condition for convergence)
    print("\nDiagonal Dominance Analysis:")
    is_dominant, diag_notes = check_diagonal_dominance(n, A)
    for note in diag_notes:
        print(f"  {note}")
    if is_dominant:
        print("  -> Matrix A is strictly diagonally dominant. Convergence is GUARANTEED.")
    else:
        print("  -> Matrix A is not strictly diagonally dominant. Convergence is not guaranteed.")

    print("\n" + "-" * 70)
    print(" Starting Jacobi Iterations:")
    print("-" * 70)

    # 4. Execute Jacobi iteration
    try:
        x_solution, total_iters, final_sn = jacobi(n, eps, A, b, x_init=x0, kmax=1000, verbose=True)
    except Exception as err:
        print(f"\nCalculation failed: {err}")
        sys.exit(1)

    # 5. Output solution vector x
    print("-" * 70)
    print(" x:")
    for i in range(n):
        print(f"  x({i + 1}) = {x_solution[i]: .10f}")

    # 6. Verification: Compute residual vector r = A*x - b
    print("\nVerification (Residual r = A*x - b):")
    residuals = []
    for i in range(n):
        ax_i = sum(A[i][j] * x_solution[j] for j in range(n))
        residuals.append(ax_i - b[i])
        print(f"  Equation {i + 1}: (A*x)_{i + 1} - b_{i + 1} = {residuals[i]: .6e}")

    max_res = max(abs(r) for r in residuals)
    print(f"\nMax absolute residual ||A*x - b||_inf = {max_res:.6e}")
    print(f"Converged successfully in {total_iters} iterations.")
    print("=" * 70)
    print(" JacobiIter: Konec vypoctu!")
    print("=" * 70)


if __name__ == "__main__":
    main()
