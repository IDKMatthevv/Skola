import math
import numpy as np

# ==============================================================================
# Task 1: Sum of 1 / n^2 (Converges to pi^2 / 6)
# ==============================================================================
print("=== ÚKOL 1: Součet 1 / n^2 ===")
exact_sum = (math.pi ** 2) / 6.0
print(f"Přesná hodnota (pi^2 / 6): {exact_sum:.15f}\n")

n_max_values = [10**3, 10**5, 10**7]

for n_max in n_max_values:
    print(f"-- n_max = {n_max:,} --")
    for dtype, label in [(np.float32, "real(4) "), (np.float64, "real(8) "), (np.longdouble, "real(16)")]:
        s = dtype(0.0)
        # Direct summation
        for n in range(1, n_max + 1):
            s += dtype(1.0) / dtype(n * n)
        
        err = abs(float(s) - exact_sum)
        print(f"  {label}: součet = {s:.10f} | chyba = {err:.3e}")
    print()


# ==============================================================================
# Task 2: Harmonic Series 1 / n (Diverges analytically)
# ==============================================================================
print("\n=== ÚKOL 2: Součet 1 / n (Harmonická řada) ===")
# We test up to 25 million to show float32 stagnation
for n_max in [10**4, 10**6, 25 * 10**6]:
    print(f"-- n_max = {n_max:,} --")
    for dtype, label in [(np.float32, "real(4) "), (np.float64, "real(8) ")]:
        s = dtype(0.0)
        for n in range(1, n_max + 1):
            prev_s = s
            s += dtype(1.0) / dtype(n)
            # Detect where addition stops changing the sum
            if n_max > 10**6 and n % 5_000_000 == 0:
                if s == prev_s:
                    print(f"  [{label}] Zastavení přičítání v n = {n:,} (1/n je menší než epsilon akumulátoru)")
        print(f"  {label}: součet = {s:.8f}")
    print()


# ==============================================================================
# Task 3: Alternating Sequence & Underflow
# ==============================================================================
print("\n=== ÚKOL 3: Posloupnost a underflow ===")
# Note: The formula for a_{2n+1} in the assignment text has a typo: it wrote "/" 
# instead of "*". For a_{2n+1} -> 1 to hold analytically, a_{2n+1} = a_{2n} * (n + 15).

for dtype, label in [(np.float32, "real(4) "), (np.float64, "real(8) ")]:
    print(f"-- Typ: {label} --")
    a_odd = dtype(1.0)      # a_1 = 1
    
    # Observe sequence up to n = 200
    underflowed = False
    for n in range(1, 200):
        # a_{2n} = a_{2n-1} / (n + 15)
        a_even = a_odd / dtype(n + 15)
        # a_{2n+1} = a_{2n} * (n + 15)
        a_odd = a_even * dtype(n + 15)
        
        if n in [1, 2, 5, 10, 35, 40, 150, 175]:
            print(f"  n = {n:3d}: a_{{2n}} = {a_even:.3e}, a_{{2n+1}} = {a_odd}")
            
        if a_even == dtype(0.0) and not underflowed:
            print(f"  --> KATASTROFA UNDERFLOW: v n = {n}, a_{{2n}} spadlo na přesnou 0.0!")
            print(f"      Proto a_{{2n+1}} = 0 * ({n} + 15) = {a_odd} (místo analytické 1.0)\n")
            underflowed = True
        break