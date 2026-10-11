import math
import matplotlib.pyplot as plt

def f(x):
    return x**3 - x - 2

def regula_falsi(a, b, tol=1e-10, max_iter=100):
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    
    history = []
    x_prev = a
    for i in range(max_iter):
        x = a - ((b - a) / (fb - fa)) * fa
        fx = f(x)
        history.append((i+1, x, fx))
        
        if abs(fx) < tol or abs(x - x_prev) < tol:
            return x, history
            
        x_prev = x
        if fa * fx < 0:
            b = x
            fb = fx
        else:
            a = x
            fa = fx
            
    return x, history

def run():
    a, b = 1.0, 2.0
    tol = 1e-10
    root, history = regula_falsi(a, b, tol)
    
    print(f"{'Iter':<5} | {'x':<15} | {'f(x)':<15}")
    print("-" * 40)
    iterations, residuals = [], []
    for it, x, fx in history:
        print(f"{it:<5} | {x:<15.10f} | {abs(fx):<15.3e}")
        iterations.append(it)
        residuals.append(abs(fx))
        
    assert abs(f(root)) < 1e-8, f"Test failed! Residual: {abs(f(root))}"
    print(f"\nNumerical test passed. Root found: {root:.10f}")
    
    plt.figure()
    plt.plot(iterations, residuals, marker='o', linestyle='-')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Residual |f(x)|')
    plt.title('Regula Falsi Convergence: f(x) = x^3 - x - 2')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run()