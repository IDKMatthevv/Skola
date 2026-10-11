import math
import cmath

def solve_cubic(a, b, c, d):
    """Solves a*x^3 + b*x^2 + c*x + d = 0."""
    if a == 0:
        raise ValueError("Coefficient 'a' cannot be zero for a cubic equation.")
    
    # Step 1: Calculate p and q
    p = (1/3) * (3 * (c/a) - (b/a)**2)
    q = (1/27) * (2 * (b/a)**3 - 9 * (b*c)/(a**2) + 27 * (d/a))
    
    # Step 2: Calculate discriminant D
    D = (p/3)**3 + (q/2)**2
    
    # Step 3 & 4: Find roots based on D
    b_term = b / (3*a)
    roots = []
    
    if D < 0:
        # Three distinct real roots using trigonometric formulation
        phi = math.acos(-q / (2 * math.sqrt(abs(p)**3 / 27)))
        y1 = 2 * math.sqrt(abs(p)/3) * math.cos(phi / 3)
        y2 = -2 * math.sqrt(abs(p)/3) * math.cos((phi + math.pi) / 3)
        y3 = -2 * math.sqrt(abs(p)/3) * math.cos((phi - math.pi) / 3)
        roots = [y1 - b_term, y2 - b_term, y3 - b_term]
    else:
        # One real, two complex roots OR three real roots with duplicates
        u_inner = -q/2 + cmath.sqrt(D)
        v_inner = -q/2 - cmath.sqrt(D)
        
        # Cube roots for complex or real numbers
        u = u_inner ** (1/3) if u_inner.real >= 0 else -(-u_inner)**(1/3)
        v = v_inner ** (1/3) if v_inner.real >= 0 else -(-v_inner)**(1/3)
        
        y1 = u + v
        y2 = -(u + v)/2 + complex(0, 1) * (math.sqrt(3)/2) * (u - v)
        y3 = -(u + v)/2 - complex(0, 1) * (math.sqrt(3)/2) * (u - v)
        
        roots = [y1 - b_term, y2 - b_term, y3 - b_term]
        
    return roots

if __name__ == "__main__":
    print("Enter coefficients for a*x^3 + b*x^2 + c*x + d = 0")
    a = float(input("a: "))
    b = float(input("b: "))
    c = float(input("c: "))
    d = float(input("d: "))
    
    roots = solve_cubic(a, b, c, d)
    print("\nRoots:")
    for i, root in enumerate(roots, 1):
        print(f"x{i} = {root:.6f}")