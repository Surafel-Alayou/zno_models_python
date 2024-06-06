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
ZnO_data_set = pd.read_csv('ZnO_dataset.csv')

# Create a separate LabelEncoder object for each categorical column
le_synthesis_method = LabelEncoder()
le_precursor = LabelEncoder()

# Fit the label encoder and transform each categorical column individually
ZnO_data_set["Synthesis method"] = le_synthesis_method.fit_transform(ZnO_data_set["Synthesis method"])
ZnO_data_set["Precursor"] = le_precursor.fit_transform(ZnO_data_set["Precursor"])

# Handling missing values by replacing them with the mean of each feature
X = ZnO_data_set.iloc[:, :-1]
X.fillna(X.mean(), inplace=True)  # Replace missing values with the mean of each feature

y = ZnO_data_set.iloc[:, -1]
y.fillna(y.mean(), inplace=True)  # Replace missing values with the mean of the target variable

# Splitting dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=SEED)

# Instantiate XGBoost Regressor
xgb_regressor = XGBRegressor(
                            learning_rate=0.022, # [0.01, 0.3]
                            n_estimators=295, # [100, inf]
                            max_depth=12, # [3, 10]
                            random_state=SEED,
                            min_child_weight=0.3,
                            subsample=0.5, # (0.0, 1.0]
                            colsample_bytree = 1.0, # [0.5, 1]
                            gamma=0.5 # [0.0, 0.5]
                            )

# Fit to training set
xgb_regressor.fit(train_X, train_y)

# Predict on test set
pred_y = xgb_regressor.predict(test_X)

# Predict on train set
train_pred_y = xgb_regressor.predict(train_X)

# Model evaluation 
print(f"r_square_for_the_test_dataset: {r2_score(test_y, pred_y):.4f}") 
print(f"maximum_error: {max_error(test_y, pred_y):.4f}")
print(f"mean_squared_error : {mean_squared_error(test_y, pred_y):.4f}") 
print(f"mean_absolute_error : {mean_absolute_error(test_y, pred_y):.4f}")
print(f"root_mean_squared_error : {mean_squared_error(test_y, pred_y, squared=False):.4f}")

print('**************************************')

# Initialize LabelEncoders for categorical variables
le_synthesis_method = LabelEncoder()
le_precursor = LabelEncoder()

# Fit the LabelEncoders with possible categories
possible_synthesis_methods = ['Sol-gel', 'Green', 'Solvothermal', 'Hydrothermal']
possible_precursors = ['Zinc nitrate', 'Zinc acetate']

le_synthesis_method.fit(possible_synthesis_methods)
le_precursor.fit(possible_precursors)

# Function to prompt user for input with handling missing values
def prompt_user_input(prompt):
    while True:
        user_input = input(prompt)
        if user_input.strip():  # Check if input is not empty after stripping whitespace
            return user_input
        else:
            print("Missing value detected. Filling with mean value.")
            return np.nan  # Return NaN for missing values

# Prompt the user for input for each feature
energy_band_gap = float(prompt_user_input("Enter Energy band gap (eV): "))
reaction_temperature = float(prompt_user_input("Enter Reaction temperature (C): "))
calcination_temperature = float(prompt_user_input("Enter Calcination temperature (C): "))
reaction_hour = float(prompt_user_input("Enter Reaction hour (hr): "))
calcination_hour = float(prompt_user_input("Enter Calcination hour (hr): "))
synthesis_method = prompt_user_input("Enter Synthesis method: ")
precursor = prompt_user_input("Enter Precursor: ")
pH = float(prompt_user_input("Enter pH: "))
precursor_concentration = float(prompt_user_input("Enter Precursor concentration (M): "))

# Create a dictionary with the provided values
data = {
    'Energy band gap (eV)': [energy_band_gap],
    'Reaction temperature (C)': [reaction_temperature],
    'Calcination temperature (C)': [calcination_temperature],
    'Reaction hour (hr)': [reaction_hour],
    'Calcination hour (hr)': [calcination_hour],
    'Synthesis method': [synthesis_method],
    'Precursor': [precursor],
    'pH': [pH],
    'Precursor concentration (M)': [precursor_concentration]
}

# Create the DataFrame
new_data = pd.DataFrame(data)

# Handle missing values by replacing them with the mean of each feature
new_data.fillna(X.mean(), inplace=True)

# If there are categorical variables, encode them
new_data["Synthesis method"] = le_synthesis_method.transform(new_data["Synthesis method"])
new_data["Precursor"] = le_precursor.transform(new_data["Precursor"])

# Predict on new data
new_predictions = xgb_regressor.predict(new_data)

print('**************************************')

print("The estimated nanoparticle size is:", new_predictions)
