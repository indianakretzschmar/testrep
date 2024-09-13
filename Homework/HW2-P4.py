import numpy as np

# Part (a) - Create the vectors t and y
t = np.arange(0, np.pi + np.pi/30, np.pi/30)  # t starts at 0, ends at π, with increments of π/30
y = np.cos(t)  # y = cos(t)

# Compute the sum S
S = np.sum(t * y)
S2 = 0
S3 = 0
for i in range(30):
    S2+=t*y
for k in S2:
    S3+=k
# Print the result
print(f"The sum is: {S}")

print(f"The sum is: {S3}")
