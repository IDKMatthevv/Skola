import numpy as np

# 32-bit signed integer test
a_32 = np.int32(2)
print("--- 32-bit Integer ---")
for i in range(1, 6):
    a_32 = a_32 * a_32
    print(f"i = {i}: {a_32}")

# 64-bit signed integer test
a_64 = np.int64(2)
print("\n--- 64-bit Integer ---")
for i in range(1, 7):
    a_64 = a_64 * a_64
    print(f"i = {i}: {a_64}")