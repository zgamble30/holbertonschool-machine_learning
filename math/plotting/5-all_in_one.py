#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Task 0 data
y0 = np.arange(0, 11) ** 3

# Task 1 data
mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

# Task 2 data
x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

# Task 3 data
x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

# Task 4 data
np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Create the figure and subplots
fig = plt.figure()
fig.suptitle('All in One', fontsize='x-small')

# Task 0: Line Graph
ax0 = plt.subplot(3, 2, 1)
ax0.plot(y0, 'r-')
ax0.set_title('Task 0 - Line Graph', fontsize='x-small')
ax0.set_xlim(0, 10)

# Task 1: Scatter Plot
ax1 = plt.subplot(3, 2, 2)
ax1.scatter(x1, y1, color='magenta')
ax1.set_title('Task 1 - Scatter Plot', fontsize='x-small')
ax1.set_xlabel('Height (in)', fontsize='x-small')
ax1.set_ylabel('Weight (lbs)', fontsize='x-small')

# Task 2: Change of Scale
ax2 = plt.subplot(3, 2, 3)
ax2.plot(x2, y2)
ax2.set_title('Task 2 - Change of Scale', fontsize='x-small')
ax2.set_xlabel('Time (years)', fontsize='x-small')
ax2.set_ylabel('Fraction Remaining', fontsize='x-small')
ax2.set_yscale('log')
ax2.set_xlim(0, 28650)

# Task 3: Two is Better Than One
ax3 = plt.subplot(3, 2, 4)
ax3.plot(x3, y31, 'r--', label='C-14')
ax3.plot(x3, y32, 'g-', label='Ra-226')
ax3.set_title('Task 3 - Two is Better Than One', fontsize='x-small')
ax3.set_xlabel('Time (years)', fontsize='x-small')
ax3.set_ylabel('Fraction Remaining', fontsize='x-small')
ax3.set_xlim(0, 20000)
ax3.set_ylim(0, 1)
ax3.legend(fontsize='x-small')

# Task 4: Frequency
ax4 = plt.subplot(3, 2, (5, 6))
ax4.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
ax4.set_title('Task 4 - Frequency', fontsize='x-small')
ax4.set_xlabel('Grades', fontsize='x-small')
ax4.set_ylabel('Number of Students', fontsize='x-small')

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()
