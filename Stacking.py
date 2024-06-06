import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
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
meta_model = LinearRegression()

# Fit the meta-model on the base model predictions
meta_model.fit(stacking_X, y_test)

# Generate final predictions using the meta-model
stacking_preds = meta_model.predict(stacking_X)

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 20,
        }

# Scatter plot of test_y against predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, stacking_preds, s=150, edgecolor='#000000', color='#3499cd', linewidths=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='#f89939', linewidth=1.6)
plt.xlabel('Actual (nm)', fontdict=font)
plt.ylabel('Predicted (nm)', fontdict=font)
plt.xticks(fontfamily='Times New Roman', fontsize=22)
plt.yticks(fontfamily='Times New Roman', fontsize=22)
r2_stack = r2_score(y_test, stacking_preds)
plt.title(f'Linear Regression (MM): R\u00b2 = {r2_stack:.4f}', fontfamily='Times New Roman', fontsize=20)
plt.legend(prop={'size': 24, 'weight': 'normal','family': 'Times New Roman'})
plt.show()

# Model evaluation 
print(f"r_square_for_the_model: {r2_score(y_test, stacking_preds):.4f}") 
print(f"maximum_error: {max_error(y_test, stacking_preds):.4f}")
print(f"mean_squared_error : {mean_squared_error(y_test, stacking_preds):.4f}") 
print(f"mean_absolute_error : {mean_absolute_error(y_test, stacking_preds):.4f}")
print(f"root_mean_squared_error : {mean_squared_error(y_test, stacking_preds, squared=False):.4f}")
