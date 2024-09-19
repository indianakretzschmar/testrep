import numpy as np

#part (a) - create vectors 
t = np.arange(0, np.pi + np.pi/30, np.pi/30)  
y = np.cos(t)  

#find s using numpy
S = np.sum(t * y)

print(f"the sum is: {S}")