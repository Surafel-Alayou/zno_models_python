import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
x_data = [
    np.array([25, 45, 65, 85, 105]),  # Varied by 20
    np.array([80, 100, 120, 140, 160]),  # Varied by 20
    np.array([80, 100, 120, 140, 160])  # Varied by 20
]
y_data = [
    np.array([17.45, 20.68, 22.51, 26.19, 25.05]),
    np.array([33.88, 35.29, 35.08, 32.59, 33.02]),
    np.array([40.33, 40.34, 40.26, 37.90, 38.29])
]
labels = ['Sample row 1', 'Sample row 2', 'Sample row 3']

# Custom font
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

plt.figure(figsize=(10, 10))

plt.xlabel("Reaction temperature (°C)", fontdict=font)
plt.ylabel("Nanoparticle size (nm)", fontdict=font)

# Plotting lines with labels in a loop
for x, y, label in zip(x_data, y_data, labels):
    plt.plot(x, y, label=label, linewidth=2.5)

# Adding legend
plt.legend(prop={'size': 22, 'weight': 'normal', 'family': 'Times New Roman'})

# Grid settings
plt.minorticks_on()
plt.xticks(fontfamily='Times New Roman', fontsize=22)
plt.yticks(fontfamily='Times New Roman', fontsize=22)
plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')

plt.show()
