import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
x_data = [
    np.array([4.01, 3.81, 3.61, 3.41, 3.21]),  # Varied by 0.2
    np.array([3.156, 3.356, 3.556, 3.756, 3.956]),  # Varied by 0.2
    np.array([2.9, 3.1, 3.3, 3.5, 3.7])  # Varied by 0.2
]
y_data = [
    np.array([17.45, 17.72, 16.11, 19.00, 29.39]),
    np.array([33.88, 29.72, 27.71, 29.75, 28.36]),
    np.array([40.33, 40.21, 28.98, 27.43, 27.86])
]
labels = ['Sample row 1', 'Sample row 2', 'Sample row 3']

# Custom font
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

plt.figure(figsize=(10, 10))

plt.xlabel("Energy band gap (eV)", fontdict=font)
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
