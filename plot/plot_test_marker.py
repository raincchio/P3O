import matplotlib.pylab as plt
import numpy as np
fmts=['-.', '-*', '-1', '-|', '-_', ]

x = np.linspace(0,100,20)
y = np.ones_like(x)
for f in fmts:


    plt.plot(x,y,f)
    y += 1

plt.legend()
plt.show()