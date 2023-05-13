#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


# Generate y values as the cube of x values in the range [0, 10]
y = np.arange(0, 11) ** 3

# Plot y values with a red line
plt.plot(y, 'r-')

# Set the x-axis limits to [0, 10]
plt.xlim(0, 10)

# Display the plot
plt.show()
