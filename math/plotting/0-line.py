#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# Plot y as a solid red line
plt.plot(y, color='red')

# Set the x-axis range to 0 to 10
plt.xlim(0, 10)

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Graph')

# Show the plot
plt.show()
