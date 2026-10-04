import math
import numpy as np

print("=== 1. exp(x) OVERFLOW ===")
# In 64-bit float, exp(x) overflows when x > ln(1.797e+308) ≈ 709.78
for x in [708.0, 709.0, 709.78, 710.0]:
    try:
        res = math.exp(x)
        print(f"math.exp({x}) = {res}")
    except OverflowError:
        print(f"math.exp({x}) -> OverflowError!")

# NumPy representation returns 'inf' instead of throwing an error:
print(f"np.exp(np.float64(710.0)) = {np.exp(np.float64(710.0))}")


print("\n=== 2. exp(-x) UNDERFLOW ===")
# exp(-x) becomes smaller than the smallest representable float, rounding to 0.0
for x in [700, 740, 745, 750]:
    res = math.exp(-x)
    print(f"math.exp(-{x}) = {res}")


print("\n=== 3. log(x) LIMIT TOWARDS ZERO ===")
# As x -> 0+, log(x) -> -inf
for x in [1e-10, 1e-100, 1e-300, 1e-323]:
    print(f"math.log({x}) = {math.log(x)}")

# What happens exactly at 0.0:
try:
    print(math.log(0.0))
except ValueError as e:
    print(f"math.log(0.0) -> ValueError: {e}")

# NumPy returns -inf with a runtime warning:
print(f"np.log(0.0) = {np.log(0.0)}")