import numpy as np

x = 9.999999995000000e-10  # Your small x value
result_expm1 = np.expm1(x)  # Using numpy's expm1 function
print(f"Result using numpy.expm1: {result_expm1}")

# Taylor series approximation for f(x) = e^x - 1
taylor_approx = x + (x**2 / 2) + (x**3 / 6) 
print(f"Result using Taylor series approximation: {taylor_approx}")

next_term = (x**3) / 6 
print(f"Size of the next term in the Taylor series: {next_term}")

