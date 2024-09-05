import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline



xvals = np.arange(1.920, 2.081, 0.001)

pvals1 = [x**9 - 18*(x**8) + 144*(x**7) - 672*(x**6) + 2016*(x**5) - 4032*(x**4) + 5376*(x**3) - 4608*(x**2) + 2304*x + 512 for x in xvals]
pvals2 = [(x-2)**9 for x in xvals]

#plt.plot(xvals, pvals1, label='Expanded form')
plt.plot(xvals, pvals2, label='(x-2)^9')
plt.legend()  
plt.grid(True)  
plt.show()
