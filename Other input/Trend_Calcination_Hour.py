import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
x_data = [
    np.array([12, 14, 16, 18, 20]),  # Varied by 2
    np.array([2, 4, 6, 8, 10]),  # Varied by 2
    np.array([12, 14, 16, 18, 20])  # Varied by 2
]
y_data = [
    np.array([17.45, 17.45, 17.45, 17.45, 17.45]),
    np.array([33.88, 37.92, 41.20, 42.62, 42.79]),
    np.array([40.33, 40.33, 40.33, 40.33, 40.33])
]
labels = ['Sample row 1', 'Sample row 2', 'Sample row 3']

# Custom font
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

plt.figure(figsize=(10, 10))

plt.xlabel("Calcination hour (hr)", fontdict=font)
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
