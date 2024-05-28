import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, r2_score   

# Setting SEED for reproducibility
SEED = 42

# Importing dataset
ZnO_data_set = pd.read_csv('Original_ZnO_dataset_2.csv')

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

# Instantiate XGBoost Regressor
xgb_regressor = XGBRegressor(
                            learning_rate=0.022, # [0.01, 0.3]
                            n_estimators=295, # [100, inf)
                            max_depth=12, # [3, inf)
                            random_state=SEED, 
                            min_child_weight=0.3, # [0.0, 1.0]
                            subsample=0.5,  # (0.0, 1.0]
                            colsample_bytree = 1.0, # [0.5, 1]
                            gamma=0.5 # [0.0, 0.5]
                            )

# Fit to training set
xgb_regressor.fit(train_X, train_y)

# Predict on test set
pred_y = xgb_regressor.predict(test_X)

# Compute feature importance
feature_importance = xgb_regressor.feature_importances_

# Get feature names
feature_names = ZnO_data_set.columns[:-1]

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

font = {'family': 'Calibri',
        'weight': 'normal',
        'size': 20,
        }

# Scatter plot of test_y against test predictions
plt.scatter(test_y, pred_y, label='Data')
m, b = np.polyfit(test_y, pred_y, 1)
plt.plot(test_y, m*test_y + b, color='red', label='Fit')
r2_test = r2_score(test_y, pred_y)
plt.title(f'XGB: R\u00b2 = {r2_test:.4f}', fontfamily='Calibri', fontsize=24)
plt.xlabel('Actual (nm)', fontdict=font)
plt.ylabel('Predicted (nm)', fontdict=font)
plt.xticks(fontfamily='Calibri', fontsize=22)
plt.yticks(fontfamily='Calibri', fontsize=22)
plt.legend(prop={'size': 24, 'weight': 'normal','family': 'Calibri'})
plt.show()

# Plot feature importance
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance', fontdict=font)
plt.xticks(fontfamily='Calibri', fontsize=22)
plt.yticks(fontfamily='Calibri', fontsize=22)
plt.title('Feature Importance', fontfamily='Calibri', fontsize=20)
plt.gca().invert_yaxis()
plt.show()

# Model evaluation 
print(f"r_square_for_the_test_dataset: {r2_score(test_y, pred_y):.4f}") 
print(f"maximum_error: {max_error(test_y, pred_y):.4f}")
print(f"mean_squared_error : {mean_squared_error(test_y, pred_y):.4f}") 
print(f"mean_absolute_error : {mean_absolute_error(test_y, pred_y):.4f}")
print(f"root_mean_squared_error : {mean_squared_error(test_y, pred_y, squared=False):.4f}")
