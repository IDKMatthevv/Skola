"""
Jacobi2.py
==========
Solves a system of linear equations A * x = b using the Jacobi Iterative Method
(Jacobiho iterační metoda).

Mathematical Background:
------------------------
For an n x n linear system A * x = b where matrix A has non-zero diagonal entries
(a_ii != 0):
Matrix A is decomposed as:
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
1-norm (sloupcová norma / column-sum norm) falls below tolerance eps:
    SN = ||x^(k) - x^(k-1)||_1 = sum_{i=1}^n |x_i^(k) - x_i^(k-1)| < eps

Convergence Condition (Podmínka konvergence):
----------------------------------------------
Jacobi's method is mathematically guaranteed to converge from any initial guess
if matrix A is strictly diagonally dominant by rows:
    |a_ii| > sum_{j != i} |a_ij|   for all i = 1, ..., n

Input File Format ('inpIterMethod.dat'):
----------------------------------------
- Line 1:       n (dimension / order of square matrix A)
- Lines 2..n+1: Matrix A column by column:
                Line 2 = Column 1: A[0][0], A[1][0], ..., A[n-1][0]
                Line 3 = Column 2: A[0][1], A[1][1], ..., A[n-1][1]
                ...
                Line n+1 = Column n: A[0][n-1], A[1][n-1], ..., A[n-1][n-1]
- Line n+2:     Vector b: b[0], b[1], ..., b[n-1]

Fortran Scientific Notation Support:
------------------------------------
Supports standard scientific floats (e.g., 1.5e-3, 2.0E+02) as well as
Fortran-style double precision notations:
  - 'd' or 'D' exponents: 4.0d0, -1.0D-02, 2.5D+03, 1.0d-6
  - Compact Fortran outputs without 'e': 1.234-02 -> 1.234e-02

Parameters:
-----------
Reads convergence tolerance 'eps' and maximum iterations 'kmax':
  - Interactively from user input (with sensible defaults when Enter is pressed),
  - Or via command-line arguments.

How to Run:
-----------
1. Interactive run (reads 'inpIterMethod.dat', prompts for eps and kmax):
       python Jacobi2.py

2. With command-line arguments:
       python Jacobi2.py inpIterMethod.dat 1e-6 1000
   or
       python Jacobi2.py 1.0d-6 1000

3. With named CLI options:
       python Jacobi2.py --file inpIterMethod.dat --eps 1.0d-6 --kmax 1000
"""

import sys
import os
import re
from typing import List, Tuple, Optional


def parse_number(token: str) -> float:
    """
    Parses a string token into a floating-point number.
    Supports standard floats (e.g., 1.5, -2.0e-3) and Fortran-style double
    precision scientific notation (e.g., 4.0d0, -1.0D-6, 1.234-02).

    Parameters:
      token : string representation of a numerical value

    Returns:
      float value of the parsed token
    """
    token = token.strip().rstrip(",")
    # Replace Fortran double precision exponent 'd'/'D' with standard 'e'
    token = token.replace("d", "e").replace("D", "e")

    # Handle compact Fortran output where 'e' is omitted before exponent sign (e.g. 1.234-04)
    if "e" not in token.lower():
        token = re.sub(r"(\d)([+-]\d+)", r"\1e\2", token)

    return float(token)


def read_input_file(filepath: str) -> Tuple[int, List[List[float]], List[float]]:
    """
    Reads order n, matrix A (stored column by column), and vector b from an external file.

    Expected File Structure:
      - Line 1:       Order n
      - Lines 2..n+1: Matrix A column by column
                      (Line j+1 gives column j, containing row entries A[0][j]..A[n-1][j])
      - Line n+2:     Right-hand side vector b

    Parameters:
      filepath : path to the data file

    Returns:
      n : matrix dimension
      A : n x n matrix as a list of rows: A[i][j] is row i, column j
      b : right-hand side vector of length n
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Input file not found: '{filepath}'")

    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # Strip comments starting with '#' or '!' and remove whitespace
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
            f"Expected at least {n + 2} data lines (1 for n, {n} for columns of A, 1 for b), "
            f"but found {len(lines)} lines in '{filepath}'."
        )

    # 2. Read matrix A column by column
    # Line j + 1 contains column j (entries for row i = 0 to n-1)
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


def check_diagonal_dominance(n: int, A: List[List[float]]) -> Tuple[bool, List[str]]:
    """
    Checks if matrix A is strictly diagonally dominant by rows:
        |a_ii| > sum_{j != i} |a_ij|   for all i = 1, ..., n

    Strict diagonal dominance is a sufficient condition guaranteeing convergence
    of the Jacobi iterative method for any starting vector x^(0).

    Parameters:
      n : matrix dimension
      A : n x n matrix

    Returns:
      is_strictly_dominant : bool indicating if strict diagonal dominance holds
      diagnostics          : list of diagnostic strings for each row
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

        diagnostics.append(
            f"Row {i + 1}: |a_{i+1}{i+1}| = {diag_val:10.4f}, "
            f"sum(|a_{i+1}j|) = {off_diag_sum:10.4f} -> {status}"
        )

    return is_strictly_dominant, diagnostics


def jacobi(
    n: int,
    eps: float,
    A: List[List[float]],
    b: List[float],
    x_init: Optional[List[float]] = None,
    kmax: int = 1000,
    verbose: bool = True
) -> Tuple[List[float], int, float]:
    """
    Solves A * x = b using the Jacobi Iteration Method.
    Directly reflects the algorithm in Fortran 95 subroutine 'Jacobi(n, eps, a, b, x)'.

    Iteration Formula:
      x_i^(k) = (1 / a_ii) * ( b_i - sum_{j != i} (a_ij * x_j^(k-1)) )

    Stopping Criterion:
      SN = ||x^(k) - x^(k-1)||_1 = sum_{i=1}^n |x_i^(k) - x_i^(k-1)| < eps

    Parameters:
      n       : Order of the matrix
      eps     : Convergence tolerance precision
      A       : n x n coefficient matrix
      b       : Right-hand side vector of length n
      x_init  : Initial iteration vector x^(0); defaults to [0, ..., 0] if None
      kmax    : Maximum number of iterations allowed
      verbose : If True, prints iteration number and norm at each step

    Returns:
      x       : Solution vector of length n
      k       : Total number of iterations performed
      SN      : Final step difference norm ||x^(k) - x^(k-1)||_1
    """
    # Verify non-zero diagonal entries: a_ii != 0
    for i in range(n):
        if abs(A[i][i]) < 1e-15:
            raise ZeroDivisionError(
                f"Zero diagonal entry encountered at A[{i+1}][{i+1}] = {A[i][i]}. "
                "Jacobi iteration requires non-zero diagonal elements."
            )

    # Initial guess vector x^(0)
    if x_init is None:
        x = [0.0] * n
    else:
        x = list(x_init)

    # x0 stores the approximation from previous iteration x^(k-1)
    x0 = list(x)

    # Iteration loop: do k = 1, kmax
    for k in range(1, kmax + 1):
        # SN = 0.0d0: Accumulator for column-sum norm (1-norm)
        SN = 0.0

        # Compute next approximation x^(k) component by component
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

            # SN = SN + dabs(x(i) - x0(i))  ! Sloupcova norma
            SN += abs(x[i] - x0[i])

        # Print iteration status
        if verbose:
            print(f" iterace = {k:4d}   sloupcova norma = {SN:.10e}")

        # if(SN < eps) exit
        if SN < eps:
            return x, k, SN

        # x0 = x (prepare for next iteration)
        x0 = list(x)

    # If loop completes without exiting, iteration count exceeded kmax
    raise RuntimeError(
        f"Jacobi: Prilis mnoho iteraci! (Exceeded maximum allowed iterations kmax = {kmax})"
    )


def resolve_input_file(requested_path: Optional[str] = None) -> str:
    """
    Resolves the input file path. If no path is specified, searches for
    'inpIterMethod.dat' in current directory, script directory, and subdirectories.
    """
    if requested_path:
        if os.path.isfile(requested_path):
            return requested_path
        raise FileNotFoundError(f"Requested input file '{requested_path}' does not exist.")

    default_name = "inpIterMethod.dat"

    # 1. Current working directory
    if os.path.isfile(default_name):
        return default_name

    # 2. Script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_script_dir = os.path.join(script_dir, default_name)
    if os.path.isfile(candidate_script_dir):
        return candidate_script_dir

    # 3. Subdirectory 'f95 codes'
    candidate_f95 = os.path.join(script_dir, "f95 codes", default_name)
    if os.path.isfile(candidate_f95):
        return candidate_f95

    # 4. Fallback to inpJacobi.dat if inpIterMethod.dat is not found
    candidate_jacobi = os.path.join(script_dir, "inpJacobi.dat")
    if os.path.isfile(candidate_jacobi):
        return candidate_jacobi

    raise FileNotFoundError(
        f"Default input file '{default_name}' not found in current directory or '{script_dir}'."
    )


def read_parameters() -> Tuple[str, float, int]:
    """
    Parses CLI arguments or interactively prompts the user for:
      - Input file path (defaults to inpIterMethod.dat)
      - Convergence tolerance precision eps (defaults to 1.0e-6)
      - Maximum number of iterations kmax (defaults to 1000)

    Supports Fortran scientific notation in CLI arguments and interactive prompts.
    """
    filepath: Optional[str] = None
    eps: Optional[float] = None
    kmax: Optional[int] = None

    args = sys.argv[1:]

    # Parse named flags if provided (--file, --eps, --kmax)
    pos_args = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-f", "--file") and i + 1 < len(args):
            filepath = args[i + 1]
            i += 2
        elif arg in ("-e", "--eps") and i + 1 < len(args):
            eps = parse_number(args[i + 1])
            i += 2
        elif arg in ("-k", "--kmax") and i + 1 < len(args):
            kmax = int(args[i + 1])
            i += 2
        elif arg.startswith("--eps="):
            eps = parse_number(arg.split("=", 1)[1])
            i += 1
        elif arg.startswith("--kmax="):
            kmax = int(arg.split("=", 1)[1])
            i += 1
        elif arg.startswith("--file="):
            filepath = arg.split("=", 1)[1]
            i += 1
        else:
            pos_args.append(arg)
            i += 1

    # Parse positional arguments
    if pos_args:
        # Check if first positional argument is a file path
        first_is_file = False
        if os.path.isfile(pos_args[0]) or pos_args[0].endswith(".dat") or pos_args[0].endswith(".txt"):
            first_is_file = True

        if first_is_file:
            if filepath is None:
                filepath = pos_args[0]
            if len(pos_args) >= 2 and eps is None:
                eps = parse_number(pos_args[1])
            if len(pos_args) >= 3 and kmax is None:
                kmax = int(pos_args[2])
        else:
            # First arg might be eps
            try:
                val = parse_number(pos_args[0])
                if eps is None:
                    eps = val
                if len(pos_args) >= 2 and kmax is None:
                    kmax = int(pos_args[1])
            except ValueError:
                if filepath is None:
                    filepath = pos_args[0]

    # Resolve input file path
    resolved_filepath = resolve_input_file(filepath)

    # Read precision eps (if not supplied via CLI)
    if eps is None:
        default_eps_str = "1.0e-6"
        prompt_eps = f"Enter precision eps [default: {default_eps_str}]: "
        try:
            user_in = input(prompt_eps).strip()
            if user_in:
                eps = parse_number(user_in)
            else:
                eps = parse_number(default_eps_str)
        except (EOFError, KeyboardInterrupt):
            print(f"\nUsing default precision eps = {default_eps_str}")
            eps = parse_number(default_eps_str)

    # Read maximum iterations kmax (if not supplied via CLI)
    if kmax is None:
        default_kmax = 1000
        prompt_kmax = f"Enter max iterations kmax [default: {default_kmax}]: "
        try:
            user_in = input(prompt_kmax).strip()
            if user_in:
                kmax = int(user_in)
            else:
                kmax = default_kmax
        except (EOFError, KeyboardInterrupt):
            print(f"\nUsing default max iterations kmax = {default_kmax}")
            kmax = default_kmax

    # Validate parameters
    if eps <= 0.0:
        raise ValueError(f"Precision eps must be positive, got {eps}")
    if kmax <= 0:
        raise ValueError(f"Max iterations kmax must be positive, got {kmax}")

    return resolved_filepath, eps, kmax


def main():
    print("=" * 70)
    print(" Jacobi Iterative Method for Linear Systems (Jacobi2.py)")
    print(" Solving A * x = b with column-by-column matrix input")
    print("=" * 70)

    # 1. Read input configuration and parameters
    try:
        filepath, eps, kmax = read_parameters()
    except Exception as err:
        print(f"Error reading parameters: {err}")
        sys.exit(1)

    print(f"\nInput file: '{filepath}'")
    print(f"Precision tolerance eps = {eps:.10e}")
    print(f"Max iterations kmax     = {kmax}")

    # 2. Read dimension n, matrix A (column by column), and vector b
    try:
        n, A, b = read_input_file(filepath)
    except Exception as err:
        print(f"Error reading input file: {err}")
        sys.exit(1)

    print(f"Matrix dimension n      = {n}")

    # 3. Display matrix A and vector b
    print("\nMatrix A (reconstructed row by row):")
    for i in range(n):
        row_str = ", ".join(f"{A[i][j]:12.6f}" for j in range(n))
        print(f"  Row {i + 1:2d}: [ {row_str} ]")

    print("\nRight-hand side vector b:")
    b_str = ", ".join(f"{val:12.6f}" for val in b)
    print(f"  b = [ {b_str} ]")

    # Initial guess x^(0) = [0, ..., 0]
    x0 = [0.0] * n
    x0_str = ", ".join(f"{val:12.6f}" for val in x0)
    print(f"\nInitial guess x^(0):")
    print(f"  x^(0) = [ {x0_str} ]")

    # 4. Check diagonal dominance (sufficient condition for convergence)
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

    # 5. Execute Jacobi iteration method
    try:
        x_solution, total_iters, final_sn = jacobi(
            n=n, eps=eps, A=A, b=b, x_init=x0, kmax=kmax, verbose=True
        )
    except Exception as err:
        print(f"\nCalculation failed: {err}")
        sys.exit(1)

    # 6. Print computed solution vector x
    print("-" * 70)
    print(" x:")
    for i in range(n):
        print(f"  x({i + 1}) = {x_solution[i]: .10f}")

    # 7. Verification: Compute residual vector r = A * x - b
    print("\nVerification (Residual r = A*x - b):")
    residuals = []
    for i in range(n):
        ax_i = sum(A[i][j] * x_solution[j] for j in range(n))
        r_i = ax_i - b[i]
        residuals.append(r_i)
        print(f"  Equation {i + 1:2d}: (A*x)_{i + 1} - b_{i + 1} = {r_i: .6e}")

    max_res = max(abs(r) for r in residuals)
    print(f"\nMax absolute residual ||A*x - b||_inf = {max_res:.6e}")
    print(f"Converged successfully in {total_iters} iterations (final step norm = {final_sn:.6e}).")
    print("=" * 70)
    print(" JacobiIter: Konec vypoctu!")
    print("=" * 70)


if __name__ == "__main__":
    main()
