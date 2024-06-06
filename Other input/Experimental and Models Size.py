import pandas as pd
import matplotlib.pyplot as plt

# Data
y1 = [17.1, 15.2, 34.24, 31.71, 31, 38.73, 26.39, 23.76, 21.5, 31.4, 39, 33, 26.53, 18.2, 24.44, 26.5, 26.17, 28, 33.5, 33, 36, 38, 42, 33.32, 35.57, 38.4, 42.63, 29.79]
y2 = [22.53, 15.78, 35.16, 27.35, 26.67, 32.01, 32.62, 22.24, 21.56, 40.11, 40.33, 32.66, 35.98, 17.45, 29.61, 27.75, 32.71, 29.94, 28.97, 30.81, 33.82, 34.48, 34.30, 33.51, 35.39, 35.96, 39.09, 33.88]

# Create DataFrame
data = {
    'Actual': y1,
    'XGBoost': y2,
}

df = pd.DataFrame(data)

# Sort DataFrame by 'Experimental Observation'
df = df.sort_values('Actual').reset_index(drop=True)

# Plotting
plt.figure(figsize=(10, 10))

for column in df.columns:
    plt.plot(df.index + 1, df[column], marker='o', linewidth=1.5, alpha=0.9, label=column)

# Custom font
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

plt.xlabel("Index", fontdict=font)
plt.ylabel("Nanoparticle size (nm)", fontdict=font)
plt.legend(prop={'size': 22, 'weight': 'normal', 'family': 'Times New Roman'})
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.xticks(fontfamily='Times New Roman', fontsize=22)
plt.yticks(fontfamily='Times New Roman', fontsize=22)
plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
plt.show()
