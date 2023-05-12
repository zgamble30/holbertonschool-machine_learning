#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# Plotting the line graph
plt.plot(x, y)

# Setting the labels and title
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of C-14')

# Setting the y-axis to logarithmic scale
plt.yscale('log')

# Setting the x-axis range
plt.xlim(0, 28650)

# Display the plot
plt.show()
