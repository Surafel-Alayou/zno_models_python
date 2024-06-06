import pandas as pd
import matplotlib.pyplot as plt

# Data
x = [3.62, 3.31, 3.17, 3.36, 3.16]
y1 = [15.2, 17.1, 26.53, 31.71, 34.24]
y2 = [15.78, 22.53, 35.98, 27.35, 35.16]
y3 = [3.10, 6.33, 3.47, 15.5, 3.38]

# Create DataFrame
data = {
    'Energy band gap': x,
    'Actual': y1,
    'XGBoost': y2,
    'Bruss': y3
}

df = pd.DataFrame(data)

# Sort DataFrame by 'Energy band gap'
df = df.sort_values('Energy band gap').reset_index(drop=True)

# Plotting
plt.figure(figsize=(10, 10))

# Custom font
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

plt.xlabel("Energy band gap", fontdict=font)
plt.ylabel("Nanoparticle size (nm)", fontdict=font)

# Plotting lines with circular markers
for column in df.columns[1:]:
    plt.plot(df['Energy band gap'], df[column], marker='o', label=column)

# Adding legend
plt.legend(prop={'size': 22, 'weight': 'normal', 'family': 'Times New Roman'})

# Grid settings
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.xticks(fontfamily='Times New Roman', fontsize=22)
plt.yticks(fontfamily='Times New Roman', fontsize=22)
plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')

plt.show()
