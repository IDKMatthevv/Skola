eps = 1.0
while (1.0 + eps / 2.0) > 1.0:
    eps /= 2.0

print(f"Calculated 64-bit Machine Epsilon: {eps}")
print(f"Verification: 1.0 + {eps} > 1.0 is {1.0 + eps > 1.0}")
print(f"Verification: 1.0 + ({eps}/2) > 1.0 is {1.0 + (eps / 2.0) > 1.0}")