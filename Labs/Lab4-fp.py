# import libraries
import numpy as np
    
def driver():

# test functions 
     f1 = lambda x: (10/(x+4))**(1/2)
# fixed point is alpha1 = 1.4987....

     f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09... 

     Nmax = 100
     tol = 1e-10

# test f1 '''
     x0 = 1.5
     x = np.zeros((0,1))
     [xstar, ier, x] = fixedpt(f1,x0,tol,Nmax)
     fit = compute_order(x[:-1], xstar)
     print('the approximate fixed point is:',xstar)
     print('f1(xstar):',f1(xstar))
     print('Error message reads:',ier)
     print("Number of iterations without Aitken:", len(x))
     print("Applying Aitken's method to fixed-point iteration:")
     x_hat = aitkens_method(f1, x0, tol, Nmax)
     print(f"Aitken's accelerated sequence: {x_hat}")
    
     # Run regular fixed point iteration for comparison
     [xstar, ier, x] = fixedpt(f1, x0, tol, Nmax)
     print(f"Fixed point iteration sequence: {x}")
    
     # Check the number of iterations and the final errors
     print(f"Iterations with Aitken: {len(x_hat)}")
     print(f"Iterations without Aitken: {len(x)}")

     # Compute and compare the order of convergence
     print("\nOrder of Convergence:")
     print("without Aitken")
     fit = compute_order(x[:-1], xstar)
     print("with Aitken")
     fit2 = compute_order(x_hat[:-1], xstar)


    # Apply Aitken’s method to accelerate convergence
     x_hat = aitkens(x)
     print("Number of iterations with Aitken:", len(x_hat))  
     print("\nConvergence check:")
     print(f"Original error (last point): {np.abs(x[-1] - xstar)}")
     print(f"Aitken accelerated error (last point): {np.abs(x_hat[-1] - xstar)}")  
#test f2 '''
     # x0 = 0.0
     # [xstar,ier] = fixedpt(f2,x0,tol,Nmax)
     # print('the approximate fixed point is:',xstar)
     # print('f2(xstar):',f2(xstar))
     # print('Error message reads:',ier)



# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    x = np.zeros((Nmax,1))
    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       #x = x.append(x,x0)
       x[count] = f(x[count-1])
       if (abs(x1-x0) <tol):
          xstar = x1        
          ier = 0
          #x = x.append(x,x1)
          print('the number of iterations:', count)

          return [xstar,ier,x[:count+1]]
       x0 = x1
    print('the number of iterations:', count)

    xstar = x1
    ier = 1
    return [xstar, ier, x]

def compute_order(x, xstar):
    diff1 = np.abs(x[1::]-xstar)
    diff2 = np.abs(x[0:-1]-xstar)

    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)

    _lambda = np.exp(fit[1])
    alpha = fit[0]
    print(f"lambda is {_lambda}")
    print(f"alpha is {alpha}")
    return fit

# def aitkens_fixedpt(f, x0, tol, Nmax):
#     x_hat = []
#     x1 = None
#     exes = []
#     for _ in range(Nmax):
#         x1 = f(x0)
#         if np.abs(x1-x0) < tol:
#             return x_hat
#         x0 = x1
#     return x_hat

def aitkens_method(f, x0, tol, Nmax):
    
    # Run fixed point iteration to get the initial sequence
    x = np.zeros((Nmax, 1))
    x[0] = x0
    for i in range(1, Nmax):
        x[i] = f(x[i-1])
        if abs(x[i] - x[i-1]) < tol:
            x = x[:i+1]
            break
    
    # Apply Aitken's ∆² method to accelerate the sequence
    n = len(x) - 2  # We need at least 3 points for Aitken's method
    if n < 1:
        raise ValueError("Aitken's method requires at least 3 iterations.")
    
    x_hat = np.zeros((n, 1))
    
    for i in range(n):
        p_n = x[i]
        p_n1 = x[i+1]
        p_n2 = x[i+2]
        
        # Apply Aitken's formula
        numerator = (p_n1 - p_n)**2
        denominator = p_n2 - 2*p_n1 + p_n
        
        if denominator != 0:
            x_hat[i] = p_n - numerator / denominator
        else:
            x_hat[i] = p_n  # No acceleration if denominator is 0
    
    return x_hat

def aitkens(x):
    
    n = len(x) - 2  # We need at least 3 points for Aitken's method to work
    if n < 1:
        raise ValueError("Aitken's method requires at least 3 iterations.")
    
    x_hat = np.zeros((n, 1))
    
    for i in range(n):
        p_n = x[i]
        p_n1 = x[i+1]
        p_n2 = x[i+2]
        
        # Apply Aitken's ∆² formula
        numerator = (p_n1 - p_n)**2
        denominator = p_n2 - 2*p_n1 + p_n
        
        if denominator != 0:
            x_hat[i] = p_n - numerator / denominator
        else:
            x_hat[i] = p_n  # No acceleration if denominator is 0

    return x_hat

driver()