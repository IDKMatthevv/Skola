import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - x - 2

def secant_method(x0, x1, tol=1e-10, max_iter=100):
    print("Secant Method Iteration Table")
    print(f"{'Iter':<5} | {'x_k':<20} | {'f(x_k)':<20} | {'Residual':<20}")
    print("-" * 75)
    
    residuals = []
    iterations = []
    
    residuals.append(abs(f(x0)))
    iterations.append(0)
    print(f"{0:<5} | {x0:<20.15f} | {f(x0):<20.15f} | {abs(f(x0)):<20.15e}")
    
    residuals.append(abs(f(x1)))
    iterations.append(1)
    print(f"{1:<5} | {x1:<20.15f} | {f(x1):<20.15f} | {abs(f(x1)):<20.15e}")
    
    x_prev = x0
    x_curr = x1
    
    for k in range(2, max_iter):
        fx_curr = f(x_curr)
        fx_prev = f(x_prev)
        res = abs(fx_curr)
        
        if res < tol:
            print("-" * 75)
            print(f"Converged in {k-1} iterations.")
            break
            
        if fx_curr - fx_prev == 0:
            print("Denominator is zero. Stopping.")
            break
            
        x_next = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        
        x_prev = x_curr
        x_curr = x_next
        
        fx_next = f(x_curr)
        residuals.append(abs(fx_next))
        iterations.append(k)
        
        print(f"{k:<5} | {x_curr:<20.15f} | {fx_next:<20.15f} | {abs(fx_next):<20.15e}")
        
    return iterations, residuals, x_curr

if __name__ == "__main__":
    print("Numerical Test: Solving x^3 - x - 2 = 0")
    x0, x1 = 1.0, 2.0
    iters, res, root = secant_method(x0, x1, tol=1e-10)
    
    plt.figure(figsize=(8, 5))
    plt.plot(iters, res, marker='s', linestyle='-', color='r')
    plt.yscale('log')
    plt.xlabel('Iteration (k)')
    plt.ylabel('Residual |f(x_k)|')
    plt.title('Secant Method Convergence: Residual vs Iteration')
    plt.grid(True, which="both", ls="--")
    plt.savefig('secant_residual.png')