import numpy as np
import matplotlib.pyplot as plt

# Define the new function
def f(x):
    return 1 / (1 + (10 * x) ** 2)

def driver():
    # Interval [-1, 1]
    a = -1
    b = 1
    # Create points to evaluate at
    Neval = 1000  # Finer grid for better resolution
    xeval = np.linspace(a, b, Neval)
    
    # Number of intervals for the linear spline
    Nint = 10  # You can try varying this for better understanding
    
    # Evaluate the linear spline
    yeval = eval_lin_spline(xeval, a, b, f, Nint)
    
    # Evaluate f at the evaluation points
    fex = f(xeval)
    
    # Plot the exact function and the spline
    plt.plot(xeval, fex, label='Exact function')
    plt.plot(xeval, yeval, label='Linear spline')
    plt.legend()
    plt.show()
    
    # Calculate the error
    err = abs(yeval - fex)
    
    # Plot the error
    plt.semilogy(xeval, err, label='Absolute error')
    plt.legend()
    plt.show()

def eval_lin_spline(xeval, a, b, f, Nint):
    # Create the intervals for piecewise approximations
    xint = np.linspace(a, b, Nint + 1)
    
    # Create vector to store the evaluation of the linear splines
    yeval = np.zeros_like(xeval)
    
    # Loop over each interval
    for jint in range(Nint):
        # Get the interval boundaries
        a1 = xint[jint]
        fa1 = f(a1)
        b1 = xint[jint + 1]
        fb1 = f(b1)
        
        # Find indices of xeval in the interval (xint[jint], xint[jint+1])
        ind = np.where((xeval >= a1) & (xeval <= b1))[0]
        
        # Linear interpolation for the points in this interval
        if len(ind) > 0:
            yeval[ind] = fa1 + (xeval[ind] - a1) * (fb1 - fa1) / (b1 - a1)
    
    return yeval

# Run the driver function
if __name__ == '__main__':
    driver()
