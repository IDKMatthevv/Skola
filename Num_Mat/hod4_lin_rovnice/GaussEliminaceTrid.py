# GaussEliminaceTrid.py
# Solves a tridiagonal system of linear equations[cite: 2].

def gauss_trid(d, p, s, b):
    n = len(b)
    
    # Forward elimination (Primy chod)[cite: 2]
    # Note: Python is 0-indexed, so we iterate from 1 to n-1
    for i in range(1, n):
        temp = p[i] / d[i - 1]
        d[i] = d[i] - temp * s[i - 1]
        b[i] = b[i] - temp * b[i - 1]
        
    # Back substitution (Zpetny chod)[cite: 2]
    b[n - 1] = b[n - 1] / d[n - 1]
    
    for i in range(n - 2, -1, -1):
        b[i] = (b[i] - s[i] * b[i + 1]) / d[i]
        
    return b

if __name__ == "__main__":
    # Hardcoded data[cite: 2]
    p = [0.0, -0.1125, -0.1125, -0.1125, -0.1125]
    d = [1.225, 1.225, 1.225, 1.225, 1.225]
    s = [-0.1125, -0.1125, -0.1125, -0.1125, 0.0]
    b = [1.307452, 0.9725676, 0.8852738, 0.8472736, 0.7876443]
    
    print("Executing Tridiagonal Gaussian Elimination...")
    result = gauss_trid(d, p, s, b)
    
    print(" x:")
    for val in result:
        print(f" {val}")
        
    print(" GaussElimTriD: Konec vypoctu! (End of calculation)")