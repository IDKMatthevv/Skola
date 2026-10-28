"""
SOR.py
======
Solves a system of linear equations A * x = b using the Successive
Over-Relaxation (SOR) method (Superrelaxační iterační metoda).

Converted and enhanced from the Fortran 95 program 'SuperRelaxIterMethod.f95'.

Mathematical Background:
------------------------
For a linear system A * x = b of dimension n with non-zero diagonal entries
(a_ii != 0), matrix A can be split into:
    A = D + L + U
where:
    - D is the diagonal matrix: D = diag(a_11, a_22, ..., a_nn)
    - L is the strictly lower triangular matrix
    - U is the strictly upper triangular matrix

In the Gauss-Seidel method (omega = 1), the solution vector is updated in-place:
    x_i^(GS) = (1 / a_ii) * [ b_i - sum_{j < i} (a_ij * x_j^(k)) - sum_{j > i} (a_ij * x_j^(k-1)) ]

The Successive Over-Relaxation (SOR) method accelerates convergence by taking
a weighted average between the previous iterate x_i^(k-1) and the Gauss-Seidel
correction x_i^(GS) using the relaxation parameter omega (omg):
    x_i^(k) = (1 - omega) * x_i^(k-1) + omega * x_i^(GS)
            = (1 - omega) * x_i^(k-1) + (omega / a_ii) * [ b_i - sum_{j < i} (a_ij * x_j^(k)) - sum_{j > i} (a_ij * x_j^(k-1)) ]

Properties of the Relaxation Parameter omega:
---------------------------------------------
- omega = 1: Standard Gauss-Seidel method.
- 0 < omega < 1: Under-relaxation (podrelaxace) - dampens oscillations, useful
  when Gauss-Seidel diverges or oscillates.
- 1 < omega < 2: Over-relaxation (nadrelaxace / superrelaxace) - accelerates convergence
  by extrapolating in the direction of the step.
- omega <= 0 or omega >= 2: Method diverges (by the Ostrowski-Reich and Kahan theorems,
  spectral radius rho(L_omega) >= |omega - 1| >= 1).

Convergence Conditions (Podmínky konvergence):
----------------------------------------------
1. Ostrowski-Reich Theorem:
   If A is a symmetric positive-definite (SPD) matrix, the SOR method converges
   for any initial guess x^(0) if and only if 0 < omega < 2.
2. Diagonally Dominant Matrices:
   If A is strictly diagonally dominant (|a_ii| > sum_{j != i} |a_ij|), SOR
   converges for 0 < omega <= 1 (Gauss-Seidel and under-relaxation). If A is also
   symmetric with positive diagonals, it is SPD, guaranteeing convergence for all 0 < omega < 2.

Stopping Criterion (Ukončovací kritérium):
------------------------------------------
The iteration stops when the 1-norm (column norm / "sloupcová norma" in the original
Fortran code) of the difference between successive iterations falls below eps:
    SN = ||x^(k) - x^(k-1)||_1 = sum_{i=1}^n |x_i^(k) - x_i^(k-1)| < eps

Input File Structure ('@inpIterMethod.dat' or 'inpIterMethod.dat'):
-------------------------------------------------------------------
- Line 1:       n (dimension / order of square matrix A)
- Lines 2..n+1: Matrix A column by column:
                Line 2 = Column 1: A[0][0], A[1][0], ..., A[n-1][0]
                Line 3 = Column 2: A[0][1], A[1][1], ..., A[n-1][1]
                ...
                Line n+1 = Column n: A[0][n-1], A[1][n-1], ..., A[n-1][n-1]
- Line n+2:     Right-hand side vector b: b[0], b[1], ..., b[n-1]

Fortran-Style Scientific Notation Support:
------------------------------------------
Supports standard scientific floats (e.g., 1.5e-3, 2.0E+02) as well as
Fortran-style double precision notations:
  - 'd' or 'D' exponents: 4.0d0, -1.0D-02, 2.5D+03, 1.0d-6
  - Compact Fortran outputs without 'e': 1.234-02 -> 1.234e-02

Parameters:
-----------
Reads:
  - omega : Relaxation parameter (default: 1.1)
  - eps   : Precision tolerance (default: 1.0e-6)
  - kmax  : Maximum iterations (default: 1000)
from interactive user input (pressing Enter accepts defaults) or CLI arguments.

How to Run:
-----------
1. Interactive run (prompts for omega, eps, and kmax):
       python SOR.py

2. Command-line positional arguments:
       python SOR.py 1.1 1.0e-6 1000
   or specifying the data file:
       python SOR.py inpIterMethod.dat 1.1 1.0e-6 1000
       python SOR.py @inpIterMethod.dat 1.1 1.0e-6 1000

3. Command-line named options:
       python SOR.py --omega 1.1 --eps 1e-6 --kmax 1000 --file @inpIterMethod.dat
"""

import os
import re
import sys
from typing import List, Optional, Tuple

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


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


def resolve_input_file(filepath: Optional[str] = None) -> str:
    """
    Resolves the input file path, checking both the current directory and the
    directory containing the script. Handles optional '@' prefix (e.g., '@inpIterMethod.dat').

    Parameters:
      filepath : Optional user-provided path or filename

    Returns:
      Absolute or valid relative path to the existing input file

    Raises:
      FileNotFoundError: If the specified or default file cannot be found
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Candidates to search if no specific filepath was given
    if filepath is None:
        candidate_names = [
            "@inpIterMethod.dat",
            "inpIterMethod.dat",
            "inpJacobi.dat",
        ]
    else:
        # Strip leading '@' if file with literal '@' doesn't exist
        raw_name = filepath.strip()
        stripped_name = raw_name[1:] if raw_name.startswith("@") else raw_name
        candidate_names = [raw_name, stripped_name]

    # Search in current directory, script directory, and as direct paths
    checked_paths = []
    for cand in candidate_names:
        direct = cand
        in_cwd = os.path.join(os.getcwd(), cand)
        in_script = os.path.join(script_dir, cand)

        for p in (direct, in_cwd, in_script):
            checked_paths.append(p)
            if os.path.isfile(p):
                return p

    raise FileNotFoundError(
        f"Could not locate input file. Checked candidate paths:\n"
        + "\n".join(f"  - {p}" for p in dict.fromkeys(checked_paths))
    )


def read_input_file(filepath: str) -> Tuple[int, List[List[float]], List[float]]:
    """
    Reads order n, matrix A (stored column by column), and vector b from an external file.

    Expected File Structure:
      - Line 1:       Order n
      - Lines 2..n+1: Matrix A column by column
                      (Line j+1 gives column j, containing row entries A[0][j]..A[n-1][j])
      - Line n+2:     Right-hand side vector b (length n)

    Parameters:
      filepath : path to the data file

    Returns:
      n : matrix dimension
      A : n x n matrix as a list of rows: A[i][j] is row i, column j
      b : right-hand side vector of length n
    """
    resolved_path = resolve_input_file(filepath)

    lines = []
    with open(resolved_path, "r", encoding="utf-8") as f:
        for line in f:
            # Strip comments starting with '#' or '!' and remove whitespace
            clean = re.sub(r"[#!].*$", "", line).strip()
            if clean:
                lines.append(clean)

    if not lines:
        raise ValueError(f"Input file '{resolved_path}' is empty or contains only comments.")

    # 1. Read dimension n
    n = int(lines[0].split()[0])
    if n <= 0:
        raise ValueError(f"Matrix dimension n must be positive, got {n}.")

    if len(lines) < n + 2:
        raise ValueError(
            f"Expected at least {n + 2} data lines (1 for n, {n} for columns of A, 1 for b), "
            f"but found {len(lines)} lines in '{resolved_path}'."
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


def sor(
    n: int,
    eps: float,
    omega: float,
    A: List[List[float]],
    b: List[float],
    x_init: Optional[List[float]] = None,
    kmax: int = 1000,
    verbose: bool = True
) -> Tuple[List[float], int, float, List[Tuple[int, float]]]:
    """
    Solves A * x = b using the Successive Over-Relaxation (SOR) method.
    Directly reflects the algorithm in Fortran 95 subroutine 'SuperRelax(n, eps, omg, a, b, x)'.

    Iteration Formula:
      For each row i = 0 to n-1:
        dsum = b[i] - sum_{j != i} (A[i][j] * x[j])
        x[i] = (1 - omega) * x0[i] + (omega / A[i][i]) * dsum
        SN  += |x[i] - x0[i]|

    Stopping Criterion:
      SN = sum_{i=0}^{n-1} |x[i] - x0[i]| < eps  (1-norm / "sloupcova norma")

    Parameters:
      n       : Order of the matrix
      eps     : Convergence tolerance
      omega   : Relaxation parameter (omg)
      A       : n x n matrix
      b       : Right-hand side vector of length n
      x_init  : Initial iteration vector x^(0); defaults to all zeros if None
      kmax    : Maximum number of iterations allowed (default: 1000)
      verbose : If True, prints iteration number and norm at each step

    Returns:
      x       : Solution vector of length n
      k       : Total number of iterations performed
      SN      : Final step norm ||x^(k) - x^(k-1)||_1
      history : List of (iteration, SN) pairs

    Raises:
      ZeroDivisionError : If a zero diagonal entry A[i][i] is encountered
      RuntimeError      : If the iteration does not converge within kmax steps
    """
    # Verify non-zero diagonal entries
    for i in range(n):
        if abs(A[i][i]) < 1e-15:
            raise ZeroDivisionError(
                f"Zero diagonal entry encountered at A[{i+1}][{i+1}] = {A[i][i]}. "
                "SOR iteration requires non-zero diagonal elements."
            )

    # Check relaxation parameter validity
    if omega <= 0.0 or omega >= 2.0:
        print(
            f"WARNING: Relaxation parameter omega = {omega} is outside the theoretical "
            f"convergence interval (0, 2). The method may diverge!"
        )

    # Počáteční iterace = 0 (Initial guess vector x^(0))
    if x_init is None:
        x = [0.0] * n
    else:
        x = list(x_init)

    # x0 = x (stores previous iteration values)
    x0 = list(x)

    history: List[Tuple[int, float]] = []

    # Iteration loop: do k = 1, kmax
    for k in range(1, kmax + 1):
        # SN = 0.0d0 ! Akumulátor pro sloupcovou normu (1-norm of increment)
        SN = 0.0

        for i in range(n):
            # dsum = b(i)
            dsum = b[i]

            # do j=1,n; if(i == j) cycle; dsum = dsum - a(i,j)*x(j); enddo
            # Notice x[j] uses newly updated values for j < i, and previous values for j > i
            for j in range(n):
                if i == j:
                    continue
                dsum -= A[i][j] * x[j]

            # x(i) = (1.0d0 - omg)*x0(i) + (omg / a(i,i))*dsum
            x[i] = (1.0 - omega) * x0[i] + (omega / A[i][i]) * dsum

            # SN = SN + dabs(x(i) - x0(i)) ! Sloupcova norma
            SN += abs(x[i] - x0[i])

        history.append((k, SN))

        # print*,"iterace =",k,"sloupcova norma =",SN
        if verbose:
            print(f" iterace = {k:4d}   sloupcova norma = {SN:.10e}")

        # if(SN < eps) exit
        if SN < eps:
            return x, k, SN, history

        # x0 = x (update previous iteration for the next step)
        x0 = list(x)

    # if(k > kmax) stop " SuperRelax: Prilis mnoho iteraci!"
    raise RuntimeError(
        f"SuperRelax: Prilis mnoho iteraci! (Maximum iterations kmax = {kmax} exceeded without reaching eps = {eps:.2e})"
    )


_piped_lines: List[str] = []
_piped_checked: bool = False


def get_user_input(prompt_text: str, default_val: str) -> str:
    """
    Safely retrieves user input:
      - In interactive terminals: prompts user via input().
      - In non-interactive environments with piped data: reads lines sequentially.
      - In non-interactive environments without waiting data: falls back to default_val.
    """
    global _piped_lines, _piped_checked

    # 1. Standard interactive console
    if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
        try:
            val = input(prompt_text).strip()
            return val if val else default_val
        except (EOFError, KeyboardInterrupt):
            print(f"\nUsing default: {default_val}")
            return default_val

    # 2. Check for waiting piped data without blocking (Windows support)
    if not _piped_checked:
        _piped_checked = True
        try:
            import msvcrt
            import ctypes
            from ctypes import wintypes
            handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            avail = wintypes.DWORD()
            success = ctypes.windll.kernel32.PeekNamedPipe(
                handle, None, 0, None, ctypes.byref(avail), None
            )
            if success and avail.value > 0:
                _piped_lines = [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]
        except Exception:
            pass

    if _piped_lines:
        return _piped_lines.pop(0)

    # 3. Non-interactive fallback when no data is waiting on stdin
    print(f"{prompt_text.strip()} [auto-default: {default_val}]")
    return default_val


def read_parameters() -> Tuple[Optional[str], float, float, int]:
    """
    Parses command-line arguments and prompts for relaxation parameter omega,
    precision eps, and maximum iterations kmax with interactive fallbacks.

    Returns:
      (filepath, omega, eps, kmax)
    """
    filepath: Optional[str] = None
    omega: Optional[float] = None
    eps: Optional[float] = None
    kmax: Optional[int] = None

    args = sys.argv[1:]
    pos_args: List[str] = []
    i = 0

    while i < len(args):
        arg = args[i]
        if arg in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        elif arg in ("-f", "--file") and i + 1 < len(args):
            filepath = args[i + 1]
            i += 2
        elif arg in ("-w", "-o", "--omega", "--omg") and i + 1 < len(args):
            omega = parse_number(args[i + 1])
            i += 2
        elif arg in ("-e", "--eps") and i + 1 < len(args):
            eps = parse_number(args[i + 1])
            i += 2
        elif arg in ("-k", "--kmax") and i + 1 < len(args):
            kmax = int(args[i + 1])
            i += 2
        elif arg.startswith("--omega="):
            omega = parse_number(arg.split("=", 1)[1])
            i += 1
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

    # Interpret positional arguments
    # Possible positional signatures:
    #   [omega]
    #   [omega, eps]
    #   [omega, eps, kmax]
    #   [filepath, omega]
    #   [filepath, omega, eps]
    #   [filepath, omega, eps, kmax]
    if pos_args:
        first = pos_args[0]
        # Check if first arg looks like a file path
        looks_like_file = (
            first.startswith("@")
            or first.endswith(".dat")
            or first.endswith(".txt")
            or os.path.isfile(first)
            or os.path.isfile(first[1:] if first.startswith("@") else first)
        )

        if looks_like_file:
            if filepath is None:
                filepath = first
            if len(pos_args) >= 2 and omega is None:
                omega = parse_number(pos_args[1])
            if len(pos_args) >= 3 and eps is None:
                eps = parse_number(pos_args[2])
            if len(pos_args) >= 4 and kmax is None:
                kmax = int(pos_args[3])
        else:
            # First positional arg is numeric omega
            try:
                val = parse_number(pos_args[0])
                if omega is None:
                    omega = val
                if len(pos_args) >= 2 and eps is None:
                    eps = parse_number(pos_args[1])
                if len(pos_args) >= 3 and kmax is None:
                    kmax = int(pos_args[2])
            except ValueError:
                if filepath is None:
                    filepath = pos_args[0]

    # Defaults
    default_omega_str = "1.1"
    default_eps_str = "1.0e-6"
    default_kmax = 1000

    # 1. Read relaxation parameter omega (if not supplied via CLI)
    if omega is None:
        raw_omg = get_user_input(f" Zadej omega [default: {default_omega_str}]: ", default_omega_str)
        omega = parse_number(raw_omg)

    # 2. Read precision eps (if not supplied via CLI)
    if eps is None:
        raw_eps = get_user_input(f" Zadej presnost eps [default: {default_eps_str}]: ", default_eps_str)
        eps = parse_number(raw_eps)

    # 3. Read maximum iterations kmax (if not supplied via CLI)
    if kmax is None:
        raw_kmax = get_user_input(f" Zadej maximalni pocet iteraci kmax [default: {default_kmax}]: ", str(default_kmax))
        kmax = int(raw_kmax)

    # Parameter validation
    if eps <= 0.0:
        raise ValueError(f"Precision eps must be positive, got {eps}")
    if kmax <= 0:
        raise ValueError(f"Max iterations kmax must be positive, got {kmax}")

    return filepath, omega, eps, kmax


def main() -> None:
    print("=" * 70)
    print(" Successive Over-Relaxation (SOR) Iterative Method (SOR.py)")
    print(" Solving A * x = b with column-by-column matrix input")
    print("=" * 70)

    # 1. Read input parameters (omega, eps, kmax) and file path
    try:
        filepath_arg, omega, eps, kmax = read_parameters()
    except Exception as err:
        print(f"Error reading parameters: {err}")
        sys.exit(1)

    # Resolve and verify input data file
    try:
        resolved_filepath = resolve_input_file(filepath_arg)
    except Exception as err:
        print(f"Error resolving input file: {err}")
        sys.exit(1)

    print(f"\nInput file:             '{resolved_filepath}'")
    print(f"Relaxation parameter w: {omega:.4f}")
    print(f"Precision tolerance eps:{eps:.10e}")
    print(f"Max iterations kmax:    {kmax}")

    # 2. Read dimension n, matrix A (column by column), and vector b
    try:
        n, A, b = read_input_file(resolved_filepath)
    except Exception as err:
        print(f"Error reading input file: {err}")
        sys.exit(1)

    print(f"Matrix dimension n:     {n}")

    # 3. Display matrix A and vector b
    print("\nMatrix A (reconstructed row by row):")
    for i in range(n):
        row_str = ", ".join(f"{A[i][j]:12.6f}" for j in range(n))
        print(f"  Row {i + 1:2d}: [ {row_str} ]")

    print("\nRight-hand side vector b:")
    b_str = ", ".join(f"{val:12.6f}" for val in b)
    print(f"  b = [ {b_str} ]")

    # Initial guess x^(0) = [0, ..., 0] (as in SuperRelaxIterMethod.f95: x = 0)
    x0 = [0.0] * n
    x0_str = ", ".join(f"{val:12.6f}" for val in x0)
    print(f"\nInitial guess x^(0):")
    print(f"  x^(0) = [ {x0_str} ]")

    # 4. Check diagonal dominance
    print("\nDiagonal Dominance Analysis:")
    is_dominant, diag_notes = check_diagonal_dominance(n, A)
    for note in diag_notes:
        print(f"  {note}")
    if is_dominant:
        print("  -> Matrix A is strictly diagonally dominant.")
    else:
        print("  -> Matrix A is not strictly diagonally dominant.")

    print("\n" + "-" * 70)
    print(" Starting SOR Iterations:")
    print("-" * 70)

    # 5. Execute SOR iteration method
    try:
        x_solution, total_iters, final_sn, _ = sor(
            n=n, eps=eps, omega=omega, A=A, b=b, x_init=x0, kmax=kmax, verbose=True
        )
    except Exception as err:
        print(f"\nCalculation failed: {err}")
        sys.exit(1)

    # 6. Print computed solution vector x
    print("-" * 70)
    print(" x:")
    for i in range(n):
        print(f"  x({i + 1}) = {x_solution[i]: .10f}")

    # Also print compact Fortran-style output
    print("\nFortran-style output:")
    for val in x_solution:
        print(f"  {val: .10f}")

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
    print(" SuperRelaxIter: Konec vypoctu!")
    print("=" * 70)


if __name__ == "__main__":
    main()
