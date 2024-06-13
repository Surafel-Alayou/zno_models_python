import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, r2_score   

# Setting SEED for reproducibility
SEED = 42

# Importing dataset
ZnO_data_set = pd.read_csv('ZnO_dataset.csv')

# Create a separate LabelEncoder object for each categorical column
le_synthesis_method = LabelEncoder()
le_precursor = LabelEncoder()

# Fit the label encoder and transform each categorical column individually
ZnO_data_set["Synthesis method"] = le_synthesis_method.fit_transform(ZnO_data_set["Synthesis method"])
ZnO_data_set["Precursor"] = le_precursor.fit_transform(ZnO_data_set["Precursor"])

# Handling missing values by replacing them with the mean of each feature
X = ZnO_data_set.iloc[:, :-1]
X.fillna(X.mean(), inplace=True)  

y = ZnO_data_set.iloc[:, -1]
y.fillna(y.mean(), inplace=True)  

# Splitting dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=SEED)

# Instantiate CatBoost Regressor
cat_regressor = CatBoostRegressor(
                            iterations=390, # [1, inf)
                            learning_rate=0.032, # [0.01, 0.2]
                            depth=4, # [1, inf)
                            l2_leaf_reg=1, # [1, inf)
                            random_strength=1.4, # [0.5, 2]
                            bagging_temperature=1.6, # [0, 2]
                            border_count=256, # [32, 64, 128,…, inf)
                            random_seed=42
                            )

# Fit to training set
cat_regressor.fit(train_X, train_y, verbose=False)

# Predict on test set
pred_y = cat_regressor.predict(test_X)

# Compute feature importance
feature_importance = cat_regressor.feature_importances_

# Get feature names
feature_names = ZnO_data_set.columns[:-1]

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 20,
        }

# Scatter plot of test_y against test predictions
plt.scatter(test_y, pred_y, s=150, edgecolor='#000000', color='#00a053', linewidths=0.6)
m, b = np.polyfit(test_y, pred_y, 1)
plt.plot(test_y, m*test_y + b, color='#febd15', linewidth=1.0)
r2_test = r2_score(test_y, pred_y)
plt.title(f'CatBoost: R\u00b2 = {r2_test:.4f}', fontfamily='Times New Roman', fontsize=24)
plt.xlabel('Actual (nm)', fontfamily='Times New Roman', fontsize=24)
plt.ylabel('Predicted (nm)', fontfamily='Times New Roman', fontsize=24)
plt.xticks(fontfamily='Times New Roman', fontsize=22)
plt.yticks(fontfamily='Times New Roman', fontsize=22)
plt.legend(prop={'size': 24, 'weight': 'normal', 'family': 'Times New Roman'})
plt.show()

# Plot feature importance
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance', fontdict=font)
plt.xticks(fontfamily='Times New Roman', fontsize=22)
plt.yticks(fontfamily='Times New Roman', fontsize=22)
plt.gca().invert_yaxis()
plt.show()

# Model evaluation 
print(f"r_square_for_the_test_dataset : {r2_score(test_y, pred_y):.4f}") 
print(f"maximum_error: {max_error(test_y, pred_y):.4f}")
print(f"mean_squared_error : {mean_squared_error(test_y, pred_y):.4f}") 
print(f"mean_absolute_error : {mean_absolute_error(test_y, pred_y):.4f}")
print(f"root_mean_squared_error : {mean_squared_error(test_y, pred_y, squared=False):.4f}")