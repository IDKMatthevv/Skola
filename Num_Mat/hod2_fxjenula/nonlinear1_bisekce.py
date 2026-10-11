import math
import matplotlib.pyplot as plt

def f(x):
    return x**3 - x - 2

def bisection(a, b, tol=1e-10, max_iter=100):
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    
    history = []
    for i in range(max_iter):
        x = (a + b) / 2.0
        fx = f(x)
        history.append((i+1, x, fx))
        
        if abs(fx) < tol or (b - a) / 2.0 < tol:
            return x, history
        
        if f(a) * fx < 0:
            b = x
        else:
            a = x
            
    return x, history

def run():
    a, b = 1.0, 2.0
    tol = 1e-10
    root, history = bisection(a, b, tol)
    
    print(f"{'Iter':<5} | {'x':<15} | {'f(x)':<15}")
    print("-" * 40)
    iterations, residuals = [], []
    for it, x, fx in history:
        print(f"{it:<5} | {x:<15.10f} | {abs(fx):<15.3e}")
        iterations.append(it)
        residuals.append(abs(fx))
        
    # Numerical Test
    assert abs(f(root)) < 1e-8, f"Test failed! Residual: {abs(f(root))}"
    print(f"\nNumerical test passed. Root found: {root:.10f}")
    
    # Plot
    plt.figure()
    plt.plot(iterations, residuals, marker='o', linestyle='-')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Residual |f(x)|')
    plt.title('Bisection Convergence: f(x) = x^3 - x - 2')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run()