import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
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


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

cat_model = CatBoostRegressor(
                            iterations=390, 
                            learning_rate=0.032, 
                            depth=4, 
                            l2_leaf_reg=1, 
                            random_strength=1.4, 
                            bagging_temperature=1.6,  
                            border_count=256, 
                            random_seed=SEED
                            )

xgb_model = XGBRegressor(
                            learning_rate=0.022, 
                            n_estimators=295, 
                            max_depth=12, 
                            random_state=SEED,
                            min_child_weight=0.3,
                            subsample=0.5, 
                            colsample_bytree = 1.0, 
                            gamma=0.5 
                            )

gb_model = GradientBoostingRegressor(
                                loss='squared_error', 
                                learning_rate=0.023, 
                                n_estimators=250, 
                                max_depth = 11, 
                                random_state = SEED,
                                max_features = 4,
                                min_samples_split=9, 
                                min_samples_leaf=1,
                                subsample=0.7,
                                criterion='friedman_mse', 
                                min_impurity_decrease=0.3, 
                                )

# Fit base models on the training data
cat_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Generate predictions from base models
cat_preds = cat_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)
gb_preds = gb_model.predict(X_test)

# Create a new dataset with base model predictions as features
stacking_X = pd.DataFrame({'cat_pred': cat_preds, 'xgb_pred': xgb_preds, 'gb_pred': gb_preds})

# Meta-model
meta_model = DecisionTreeRegressor()

# Fit the meta-model on the base model predictions
meta_model.fit(stacking_X, y_test)

# Generate final predictions using the meta-model
stacking_preds = meta_model.predict(stacking_X)

# Model evaluation 
print(f"r_square_for_the_model: {r2_score(y_test, stacking_preds):.4f}") 
print(f"maximum_error: {max_error(y_test, stacking_preds):.4f}")
print(f"mean_squared_error : {mean_squared_error(y_test, stacking_preds):.4f}") 
print(f"mean_absolute_error : {mean_absolute_error(y_test, stacking_preds):.4f}")
print(f"root_mean_squared_error : {mean_squared_error(y_test, stacking_preds, squared=False):.4f}")

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

# Generate predictions from base models on new data
cat_new_preds = cat_model.predict(new_data)
xgb_new_preds = xgb_model.predict(new_data)
gb_new_preds = gb_model.predict(new_data)

# Create a new dataset with base model predictions as features
new_stacking_X = pd.DataFrame({'cat_pred': cat_new_preds, 'xgb_pred': xgb_new_preds, 'gb_pred': gb_new_preds})

# Predict using the meta-model
new_predictions = meta_model.predict(new_stacking_X)

print('**************************************')

print("The estimated nanoparticle size is:", new_predictions)