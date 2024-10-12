#libraries:
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm
import time


def driver():
    Nmax = 100
    x0 = np.array([0, 0, 1])  # Initial guess
    tol = 1e-6


    print("\nNewton's Method:")
    x_newton, g_newton, ier_newton = NewtonMethod(x0, tol, Nmax)
    print(f"Newton's Method solution: {x_newton}, gval: {g_newton}, ier: {ier_newton}")

    print("\nSteepest Descent Method:")
    x_sd, g_sd, ier_sd = SteepestDescent(x0, tol, Nmax)
    print(f"Steepest Descent solution: {x_sd}, gval: {g_sd}, ier: {ier_sd}")


    print("\nHybrid Method (Steepest Descent followed by Newton's):")
    x_hybrid, g_hybrid, ier_hybrid = hybrid_method(x0)
    print(f"Hybrid method solution: {x_hybrid}, gval: {g_hybrid}, ier: {ier_hybrid}")


def NewtonMethod(x0, tol=1e-6, Nmax=100):
    x = np.array(x0)
    
    for its in range(Nmax):
        F_val = evalF(x)
        J_val = evalJ(x)
        
        # Solve for Newton step
        try:
            delta_x = np.linalg.solve(J_val, -F_val)
        except np.linalg.LinAlgError:
            print("Jacobian is singular, unable to proceed.")
            return x, np.inf, 1
        
        x = x + delta_x
        
        # Check for convergence
        if norm(delta_x) < tol:
            return x, norm(evalF(x)), 0  # Converged
        
    print("Max iterations exceeded in Newton's method.")
    return x, norm(evalF(x)), 1  # Did not converge



def hybrid_method(x0, tol_steepest=5e-2, tol_newton=1e-6, Nmax=100):
    # First use Steepest Descent to find an initial approximation
    x_approx, gval, ier = SteepestDescent(x0, tol_steepest, Nmax)
    
    if ier == 0:
        print(f"Steepest Descent converged to {x_approx} with gval = {gval}")
    else:
        print("Steepest Descent did not converge.")
        return x_approx, gval, ier
    
    # Now use Newton's method with the result from Steepest Descent as the initial guess
    x_final, gval_final, ier_newton = NewtonMethod(x_approx, tol_newton, Nmax)
    
    return x_final, gval_final, ier_newton



###########################################################
#functions:
def evalF(x):

    F = np.zeros(3)
    F[0] = x[0] +math.cos(x[0]*x[1]*x[2])-1.
    F[1] = (1.-x[0])**(0.25) + x[1] +0.05*x[2]**2 -0.15*x[2]-1
    F[2] = -x[0]**2-0.1*x[1]**2 +0.01*x[1]+x[2] -1
    return F

def evalJ(x): 

    J =np.array([[1.+x[1]*x[2]*math.sin(x[0]*x[1]*x[2]),x[0]*x[2]*math.sin(x[0]*x[1]*x[2]),x[1]*x[0]*math.sin(x[0]*x[1]*x[2])],
          [-0.25*(1-x[0])**(-0.75),1,0.1*x[2]-0.15],
          [-2*x[0],-0.2*x[1]+0.01,1]])
    return J

def evalg(x):

    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
    F = evalF(x)
    J = evalJ(x)
    
    gradg = np.transpose(J).dot(F)
    return gradg


###############################
### steepest descent code

def SteepestDescent(x,tol,Nmax):
    
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier]



if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()