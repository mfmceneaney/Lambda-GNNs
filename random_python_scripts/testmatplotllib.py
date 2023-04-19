import matplotlib.pyplot as plt
import numpy as np

figsize = (16,10)
for style in plt.style.available:

    f = plt.figure(figsize=figsize)
    plt.title(style)
    with plt.style.context('dark_background'):
        plt.plot(np.sin(np.linspace(0, 2 * np.pi)), 'r-o')
plt.show()
print("DONE")