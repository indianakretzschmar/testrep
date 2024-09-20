import numpy as np
import matplotlib.pyplot as plt

xvals = np.linspace(-3,5,200)
yvals = [x - 4*np.sin(2*x) - 3 for x in xvals]

plt.plot(xvals, yvals)
plt.title("Plot of $f(x) = x - 4 \sin(2x) - 3$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid(True)
plt.legend()
plt.show()
