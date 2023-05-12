#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# Define the colors for each fruit
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# Plot the stacked bar graph
plt.bar(np.arange(3), fruit[0], width=0.5, color=colors[0], label='apples')
for i in range(1, 4):
    plt.bar(np.arange(3), fruit[i], width=0.5, bottom=np.sum(fruit[:i], axis=0), color=colors[i], label=['bananas', 'oranges', 'peaches'][i-1])

# Set the labels, title, and ticks
plt.xlabel('Person')
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.xticks(np.arange(3), ['Farrah', 'Fred', 'Felicia'])
plt.yticks(np.arange(0, 81, 10))
plt.ylim((0, 80))

# Add the legend
plt.legend()

# Display the plot
plt.show()
