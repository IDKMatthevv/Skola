"""
GaussEliminaceTrid1.py
======================
Solves a system of linear equations A * x = b with a tridiagonal matrix
using the Thomas algorithm (Tridiagonal Matrix Algorithm - TDMA).

The algorithm performs direct Gaussian elimination and back substitution
explicitly on the diagonal vectors without constructing the full n x n matrix,
achieving optimal O(n) computational time and O(n) memory storage.

Tridiagonal Matrix Representation:
-----------------------------------
A system of order n:
    [ d_1  s_1   0    0   ...   0   ] [ x_1 ]   [ b_1 ]
    [ p_2  d_2  s_2   0   ...   0   ] [ x_2 ]   [ b_2 ]
    [  0   p_3  d_3  s_3  ...   0   ] [ x_3 ] = [ b_3 ]
    [ ... ...  ...  ...  ...   ...  ] [ ... ]   [ ... ]
    [  0   ...   0  p_n  d_n-1 s_n-1] [x_n-1]   [b_n-1]
    [  0   ...   0    0   p_n   d_n ] [ x_n ]   [ b_n ]

Vectors (1-based math notation):
  - p : lower subdiagonal, entries p_2 ... p_n (p_1 = 0)
  - d : main diagonal,     entries d_1 ... d_n
  - s : upper superdiagonal, entries s_1 ... s_n-1 (s_n = 0)
  - b : right-hand side vector, entries b_1 ... b_n

Pivoting Note:
--------------
The Thomas algorithm does NOT employ row pivoting (partial pivoting).
In general Gaussian elimination, rows are swapped to select the largest pivot
and prevent division by zero or numerical instability. For tridiagonal systems,
row swapping would introduce non-zero fill-in elements outside the tridiagonal
bands, destroying the tridiagonal structure and increasing computational complexity
from O(n) to O(n^2).
Therefore, the algorithm assumes the matrix is non-singular and well-conditioned
(e.g., strictly diagonally dominant or symmetric positive-definite).
Pivot elements are checked before every division, and an explicit ZeroDivisionError
is raised if a zero or near-zero pivot is encountered.

Input File Format ('inpGaussEliminaceTrid.dat'):
------------------------------------------------
- Line 1: Order n (integer)
- Line 2: Lower subdiagonal vector p (n entries with p_1 = 0, or n-1 entries p_2..p_n)
- Line 3: Main diagonal vector d (n entries d_1..d_n)
- Line 4: Upper superdiagonal vector s (n entries with s_n = 0, or n-1 entries s_1..s_n-1)
- Line 5: Right-hand side vector b (n entries b_1..b_n)

Scientific notation such as 1.5e-3 and Fortran-style double precision (1.225d0)
are both supported. Comments starting with '#' or '!' are ignored.

How to Run:
-----------
1. Default run (reads 'inpGaussEliminaceTrid.dat'):
       python GaussEliminaceTrid1.py

2. Custom input file:
       python GaussEliminaceTrid1.py path/to/custom_file.dat
"""

import sys
import os
import re


def parse_number(token: str) -> float:
    """
    Parses a string token into a float.
    Supports standard scientific notation (1.5e-3, 2.0E+2)
    and Fortran double precision notation (1.225d0, -0.1125D0).
    """
    token = token.strip().rstrip(",")
    # Replace Fortran exponent indicators 'd' or 'D' with 'e'
    token_clean = token.replace("d", "e").replace("D", "e")
    return float(token_clean)


def read_input_file(filepath: str):
    """
    Reads order n, lower diagonal p, main diagonal d, upper diagonal s,
    and right-hand side vector b from the specified input file.

    Parameters:
      filepath : path to input file

    Returns:
      n : dimension of the system (int)
      p : lower subdiagonal list of length n (p[0] = 0.0)
      d : main diagonal list of length n
      s : upper superdiagonal list of length n (s[n-1] = 0.0)
      b : right-hand side list of length n
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

    # 1. Read dimension n from the first logical line
    first_tokens = lines[0].replace(",", " ").split()
    n = int(first_tokens[0])
    if n <= 0:
        raise ValueError(f"Dimension n must be a positive integer, got {n}.")

    # 2. Check if we have 4 subsequent vector lines
    # (p line, d line, s line, b line)
    if len(lines) == 5:
        p_tokens = lines[1].replace(",", " ").split()
        d_tokens = lines[2].replace(",", " ").split()
        s_tokens = lines[3].replace(",", " ").split()
        b_tokens = lines[4].replace(",", " ").split()

        p_vals = [parse_number(t) for t in p_tokens]
        d_vals = [parse_number(t) for t in d_tokens]
        s_vals = [parse_number(t) for t in s_tokens]
        b_vals = [parse_number(t) for t in b_tokens]
    else:
        # Fallback: flatten all tokens after line 0
        all_tokens = []
        for line in lines[1:]:
            all_tokens.extend(line.replace(",", " ").split())

        # Expected token counts:
        # Full representation: n for p, n for d, n for s, n for b -> 4 * n tokens
        # Reduced representation: n-1 for p, n for d, n-1 for s, n for b -> 4 * n - 2 tokens
        if len(all_tokens) == 4 * n:
            p_vals = [parse_number(t) for t in all_tokens[0:n]]
            d_vals = [parse_number(t) for t in all_tokens[n:2*n]]
            s_vals = [parse_number(t) for t in all_tokens[2*n:3*n]]
            b_vals = [parse_number(t) for t in all_tokens[3*n:4*n]]
        elif len(all_tokens) == 4 * n - 2:
            p_vals = [0.0] + [parse_number(t) for t in all_tokens[0:n-1]]
            d_vals = [parse_number(t) for t in all_tokens[n-1:2*n-1]]
            s_vals = [parse_number(t) for t in all_tokens[2*n-1:3*n-2]] + [0.0]
            b_vals = [parse_number(t) for t in all_tokens[3*n-2:4*n-2]]
        else:
            raise ValueError(
                f"Unexpected number of values in file ({len(all_tokens)} values for n = {n}). "
                f"Expected either 4 lines (p, d, s, b) or {4 * n} / {4 * n - 2} numerical entries."
            )

    # Validate and standardize p (length n, p[0] == 0.0)
    if len(p_vals) == n - 1:
        p = [0.0] + p_vals
    elif len(p_vals) == n:
        p = list(p_vals)
        if abs(p[0]) > 1e-15:
            # lower diagonal has no entry on row 1
            p[0] = 0.0
    else:
        raise ValueError(f"Lower diagonal p has {len(p_vals)} entries; expected {n} or {n - 1}.")

    # Validate d (length n)
    if len(d_vals) != n:
        raise ValueError(f"Main diagonal d has {len(d_vals)} entries; expected {n}.")
    d = list(d_vals)

    # Validate and standardize s (length n, s[n-1] == 0.0)
    if len(s_vals) == n - 1:
        s = list(s_vals) + [0.0]
    elif len(s_vals) == n:
        s = list(s_vals)
        if abs(s[n - 1]) > 1e-15:
            # upper diagonal has no entry on row n
            s[n - 1] = 0.0
    else:
        raise ValueError(f"Upper diagonal s has {len(s_vals)} entries; expected {n} or {n - 1}.")

    # Validate b (length n)
    if len(b_vals) != n:
        raise ValueError(f"Right-hand side vector b has {len(b_vals)} entries; expected {n}.")
    b = list(b_vals)

    return n, p, d, s, b


def thomas_algorithm(n: int, p_in: list, d_in: list, s_in: list, b_in: list, tol: float = 1e-15):
    """
    Solves the tridiagonal linear system A * x = b using the Thomas algorithm (TDMA).
    
    Operates directly on the 1D diagonal vectors without creating any full 2D matrix.
    
    Parameters:
      n    : order of the system (int)
      p_in : lower subdiagonal vector (length n, p_in[0] = 0)
      d_in : main diagonal vector (length n)
      s_in : upper superdiagonal vector (length n, s_in[n-1] = 0)
      b_in : right-hand side vector (length n)
      tol  : pivot zero-threshold (default 1e-15)
      
    Returns:
      x : solution vector (list of length n)
      
    Raises:
      ZeroDivisionError : if a zero or near-zero pivot is encountered.
                          Note: The Thomas algorithm does NOT perform pivoting.
    """
    if n == 1:
        if abs(d_in[0]) < tol:
            raise ZeroDivisionError(
                f"Zero pivot encountered at d[1] = {d_in[0]:.6e}. Cannot solve 1x1 system."
            )
        return [b_in[0] / d_in[0]]

    # Work on copies to keep original vectors intact
    d = list(d_in)
    b = list(b_in)
    p = list(p_in)
    s = list(s_in)

    # =========================================================================
    # 1. Forward Elimination (Přímý chod)
    # =========================================================================
    # Eliminates lower subdiagonal entries p[i] for i = 1 .. n-1 (math indices 2 .. n)
    # Note: No row pivoting is performed in the Thomas algorithm.
    for i in range(1, n):
        pivot = d[i - 1]
        if abs(pivot) < tol:
            raise ZeroDivisionError(
                f"Zero or near-zero pivot encountered at d[{i}] = {pivot:.6e} "
                f"during forward elimination (step i={i + 1}).\n"
                "Explanation: The Thomas algorithm (TDMA) does not use row pivoting (partial pivoting)\n"
                "because row swapping would destroy the tridiagonal structure and increase complexity\n"
                "from O(n) to O(n^2). Gaussian elimination cannot proceed without pivoting."
            )

        factor = p[i] / pivot
        d[i] = d[i] - factor * s[i - 1]
        b[i] = b[i] - factor * b[i - 1]
        p[i] = 0.0  # Lower diagonal entry is eliminated

    # =========================================================================
    # 2. Back Substitution (Zpětný chod)
    # =========================================================================
    # Check the last diagonal pivot d[n-1] before starting back substitution
    if abs(d[n - 1]) < tol:
        raise ZeroDivisionError(
            f"Zero or near-zero pivot encountered at d[{n}] = {d[n - 1]:.6e} "
            f"during back substitution.\n"
            "The system is singular or ill-conditioned. Back substitution cannot proceed."
        )

    x = [0.0] * n

    # Solve for the last component x[n-1] (math x_n)
    x[n - 1] = b[n - 1] / d[n - 1]

    # Compute remaining components from n-2 down to 0 (math x_{n-1} down to x_1)
    for i in range(n - 2, -1, -1):
        pivot = d[i]
        if abs(pivot) < tol:
            raise ZeroDivisionError(
                f"Zero or near-zero pivot encountered at d[{i + 1}] = {pivot:.6e} "
                f"during back substitution (step i={i + 1}).\n"
                "Division by zero prevented."
            )
        x[i] = (b[i] - s[i] * x[i + 1]) / pivot

    return x


def compute_residual(n: int, p: list, d: list, s: list, b: list, x: list) -> list:
    """
    Computes the residual vector r = A * x - b directly using the
    diagonal vectors p, d, s without constructing the full n x n matrix.
    
    r_i = p_i * x_{i-1} + d_i * x_i + s_i * x_{i+1} - b_i
    """
    residual = [0.0] * n
    for i in range(n):
        ax_i = d[i] * x[i]
        if i > 0:
            ax_i += p[i] * x[i - 1]
        if i < n - 1:
            ax_i += s[i] * x[i + 1]
        residual[i] = ax_i - b[i]
    return residual


def main():
    # Resolve input file path: command-line argument or default
    default_filename = "inpGaussEliminaceTrid.dat"

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Check current working directory first, then script's directory
        candidate_cwd = default_filename
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
        candidate_script = os.path.join(script_dir, default_filename)

        if os.path.isfile(candidate_cwd):
            filepath = candidate_cwd
        elif os.path.isfile(candidate_script):
            filepath = candidate_script
        else:
            filepath = candidate_cwd

    print("=" * 70)
    print(" Thomas Algorithm for Tridiagonal System Ax = b (GaussEliminaceTrid1)")
    print("=" * 70)
    print(f"Reading input from: {filepath}\n")

    try:
        n, p, d, s, b = read_input_file(filepath)
    except Exception as err:
        print(f"Error reading input file: {err}")
        sys.exit(1)

    print(f"Matrix dimension n = {n}")
    print("\nTridiagonal vectors (1-based math indexing):")
    print(f"  {'i':>4} | {'p(i) [lower]':>16} | {'d(i) [main]':>16} | {'s(i) [upper]':>16} | {'b(i) [rhs]':>16}")
    print("  " + "-" * 74)
    for i in range(n):
        p_str = f"{p[i]:16.6f}" if i > 0 else f"{'(p_1 = 0)':>16}"
        s_str = f"{s[i]:16.6f}" if i < n - 1 else f"{'(s_n = 0)':>16}"
        print(f"  {i + 1:4d} | {p_str} | {d[i]:16.6f} | {s_str} | {b[i]:16.6f}")

    # Solve using the Thomas algorithm
    try:
        x = thomas_algorithm(n, p, d, s, b)
    except ZeroDivisionError as err:
        print(f"\nComputation failed:\n{err}")
        sys.exit(1)

    # Print the solution vector x
    print("\n" + "=" * 70)
    print(" x:")
    for i in range(n):
        print(f"  x({i + 1}) = {x[i]: .10f}")
    print("=" * 70)

    # Verify solution by computing residual r = A*x - b without full matrix
    residual = compute_residual(n, p, d, s, b, x)
    max_residual = max(abs(r) for r in residual)
    print(f"\nVerification max |(Ax - b)_i| = {max_residual:.3e}")
    if max_residual < 1e-10:
        print("Status: Solution is accurate (residual is below tolerance).")
    else:
        print("Status: Caution, residual is larger than 1e-10.")


if __name__ == "__main__":
    main()
