# GaussEliminace.py
# Solves a system of linear equations using Gaussian Elimination[cite: 1].

def gauss(a, b):
    n = len(b)
    x = [0.0] * n
    
    # Forward elimination (Primy chod)[cite: 1]
    for k in range(n - 1):
        for i in range(k + 1, n):
            temp = a[i][k] / a[k][k]
            
            for j in range(k + 1, n):
                a[i][j] = a[i][j] - temp * a[k][j]
                
            b[i] = b[i] - temp * b[k]
            
    # Back substitution (Zpetny chod)[cite: 1]
    x[n - 1] = b[n - 1] / a[n - 1][n - 1]
    
    for i in range(n - 2, -1, -1):
        tsum = 0.0
        for j in range(i + 1, n):
            tsum += a[i][j] * x[j]
        x[i] = (b[i] - tsum) / a[i][i]
        
    return x

if __name__ == "__main__":
    # Hardcoded data equivalent to the Fortran RESHAPE setup[cite: 1]
    a = [
        [1.0, 4.0, 9.0],
        [2.0, 5.0, 6.0],
        [3.0, 6.0, 5.0]
    ]
    b = [14.0, 32.0, 36.0]
    
    print("Executing Gaussian Elimination...")
    x = gauss(a, b)
    
    print(" x:")
    for val in x:
        print(f" {val}")
        
    print(" GaussElim: Konec vypoctu! (End of calculation)")