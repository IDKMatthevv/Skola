import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def f(x):
    return x**3 - x - 2

def generate_regula_falsi_data(a, b, tol=1e-10, max_iter=20):
    """Generates the state data for each iteration to feed the animation."""
    history = []
    fa, fb = f(a), f(b)
    
    if fa * fb >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
        
    for _ in range(max_iter):
        # Calculate new x intercept (Regula Falsi formula)
        x = a - ((b - a) / (fb - fa)) * fa
        fx = f(x)
        
        # Save the current state for the frame
        history.append((a, b, fa, fb, x, fx))
        
        if abs(fx) < tol:
            break
            
        # Update boundaries
        if fa * fx < 0:
            b = x
            fb = fx
        else:
            a = x
            fa = fx
            
    return history

def run_animation():
    # 1. Generate the data
    a, b = 1.0, 2.0
    history = generate_regula_falsi_data(a, b)
    
    # 2. Setup the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0.8, 2.2)
    ax.set_ylim(-2.5, 5.0)
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    
    # Plot the static function curve
    xs = np.linspace(0.8, 2.2, 400)
    ys = f(xs)
    ax.plot(xs, ys, 'k-', linewidth=2, label='f(x) = x³ - x - 2')
    
    # Initialize animated elements
    secant_line, = ax.plot([], [], 'r--', linewidth=1.5, label='Secant Line')
    bound_points, = ax.plot([], [], 'bo', markersize=8, label='Boundaries [a, b]')
    new_x_point, = ax.plot([], [], 'go', markersize=8, label='New x guess')
    proj_line, = ax.plot([], [], 'g:', linewidth=1.5) # Line from x to f(x)
    
    ax.legend(loc='upper left')

    # 3. Animation update function
    def animate(frame_idx):
        a_curr, b_curr, fa_curr, fb_curr, x_curr, fx_curr = history[frame_idx]
        
        # Update the secant line connecting (a, f(a)) and (b, f(b))
        secant_line.set_data([a_curr, b_curr], [fa_curr, fb_curr])
        
        # Update the boundary points
        bound_points.set_data([a_curr, b_curr], [fa_curr, fb_curr])
        
        # Update the new x intercept point on the x-axis
        new_x_point.set_data([x_curr], [0])
        
        # Update the projection line showing f(x) for the new point
        proj_line.set_data([x_curr, x_curr], [0, fx_curr])
        
        ax.set_title(f'Regula Falsi - Iteration {frame_idx + 1}\n'
                     f'a={a_curr:.4f}, b={b_curr:.4f}  →  x={x_curr:.6f}')
        
        return secant_line, bound_points, new_x_point, proj_line

    # 4. Create and run the animation
    # interval=1500 means 1.5 seconds per frame so you can see the updates
    ani = animation.FuncAnimation(
        fig, animate, frames=len(history), 
        interval=1500, blit=True, repeat=False
    )
    
    plt.show()

if __name__ == "__main__":
    run_animation()