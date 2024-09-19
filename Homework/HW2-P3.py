import numpy as np

def fun(x):
    y=np.e**(x)
    print(y-1)
    return y-1

x=9.999999995000000*10**(-10)
fun(x)

