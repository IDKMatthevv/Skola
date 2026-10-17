import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - x - 2

def df(x):
    return 3*x**2 - 1

def newton_raphson(x0, tol=1e-10, max_iter=100):
    print("Newton-Raphson Method Iteration Table")
    print(f"{'Iter':<5} | {'x_k':<20} | {'f(x_k)':<20} | {'Residual':<20}")
    print("-" * 75)
    
    x = x0
    residuals = []
    iterations = []
    
    for k in range(max_iter):
        fx = f(x)
        res = abs(fx)
        residuals.append(res)
        iterations.append(k)
        
        print(f"{k:<5} | {x:<20.15f} | {fx:<20.15f} | {res:<20.15e}")
        
        if res < tol:
            print("-" * 75)
            print(f"Converged in {k} iterations.")
            break
            
        dfx = df(x)
        if dfx == 0:
            print("Derivative is zero. Stopping.")
            break
            
        x = x - fx / dfx
        
    return iterations, residuals, x

if __name__ == "__main__":
    print("Numerical Test: Solving x^3 - x - 2 = 0")
    x0 = 1.5
    iters, res, root = newton_raphson(x0, tol=1e-10)
    
    plt.figure(figsize=(8, 5))
    plt.plot(iters, res, marker='o', linestyle='-', color='b')
    plt.yscale('log')
    plt.xlabel('Iteration (k)')
    plt.ylabel('Residual |f(x_k)|')
    plt.title('Newton-Raphson Convergence: Residual vs Iteration')
    plt.grid(True, which="both", ls="--")
    plt.savefig('newton_residual.png')