import numpy as np
def compute_order(x, xstar):
    diff1 = np.abs(x[1::]-xstar)
    diff2 = np.abs(x[0:-1]-xstar)

    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)

    _lambda = np.exp(fit[1])
    alpha = fit[0]
    print(f"lambda is {_lambda}")
    print(f"alpha is {alpha}")
    return fit

    