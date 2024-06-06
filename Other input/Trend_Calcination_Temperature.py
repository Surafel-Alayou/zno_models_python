import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
x_data = [
    np.array([70, 170, 270, 370, 470]),  # Varied by 100
    np.array([600, 500, 400, 300, 200]),  # Varied by 100
    np.array([100, 200, 300, 400, 500])  # Varied by 100
]
y_data = [
    np.array([17.45, 18.23, 18.31, 19.32, 18.99]),
    np.array([33.88, 35.35, 28.95, 29.89, 29.03]),
    np.array([40.33, 39.75, 39.64, 38.69, 41.89])
]
labels = ['Sample row 1', 'Sample row 2', 'Sample row 3']

# Custom font
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

plt.figure(figsize=(10, 10))

plt.xlabel("Calcination temperature (°C)", fontdict=font)
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
