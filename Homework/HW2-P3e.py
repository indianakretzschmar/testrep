import numpy as np

x = 9.999999995000000e-10 

#using numpys expm1 function
result_expm1 = np.expm1(x) 
print(f"Result using expm1: {result_expm1}")

#taylor series approximation 
taylor_approx = x + (x**2 / 2) + (x**3 / 6) 
print(f"Result using Taylor series: {taylor_approx}")

#finding next term for double check
next_term = (x**3) / 6 
print(f"Size of term: {next_term}")

