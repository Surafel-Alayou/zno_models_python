import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, r2_score   

# Assuming xgb_regressor is your trained XGBoost model and ZnO_data_set is your dataset

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

# Instantiate XGBoost Regressor
xgb_regressor = XGBRegressor(
                            learning_rate=0.022, 
                            n_estimators=295, 
                            max_depth=12,
                            random_state=SEED,
                            min_child_weight=0.3,
                            subsample=0.5, 
                            colsample_bytree = 1.0, 
                            gamma=0.5 
                            )

# Fit to training set
xgb_regressor.fit(train_X, train_y)

# Predict on test set
pred_y = xgb_regressor.predict(test_X)

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 20,
        }

# Compute SHAP values
explainer = shap.Explainer(xgb_regressor)
shap_values = explainer.shap_values(ZnO_data_set.iloc[:, :-1])

# Average the SHAP values across all samples for each feature
mean_shap_values = np.mean(shap_values, axis=0)

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': ZnO_data_set.columns[:-1],
    'SHAP Value': mean_shap_values,
    'Importance': np.abs(mean_shap_values)  # Use absolute values for sorting
})

# Sort the DataFrame by absolute SHAP values
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 8))
colors = importance_df['SHAP Value'].apply(lambda x: 'red' if x < 0 else 'blue')
plt.barh(importance_df['Feature'], importance_df['SHAP Value'], color=colors)
plt.xlabel('Impact on target', fontdict=font)
plt.xticks(fontfamily='Times New Roman', fontsize=22)
plt.yticks(fontfamily='Times New Roman', fontsize=22)
plt.gca().invert_yaxis()
plt.show()