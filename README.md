# zno_models_python

This repository contains machine learning models developed to predict the sizes of ZnO nanoparticles based on nine features. These models include CatBoost, Gradient Boosting, Extreme Gradient Boosting, and Random Forest Regressors. The features used for prediction are:

- Energy Band Gap (eV)
- Reaction Temperature (°C)
- Calcination Temperature (°C)
- Reaction Duration (hr)
- Calcination Duration (hr)
- Synthesis Method
- Precursor
- pH
- Precursor Concentration (M)

## Features

- **Machine Learning Models**: CatBoost, Gradient Boosting, Extreme Gradient Boosting, Random Forest Regressors.
- **Hyperparameter Tuning**: Each model's hyperparameters have been meticulously tuned to achieve the best performance.
- **Performance Evaluation**: The models have been evaluated using several metrics, with Extreme Gradient Boosting showing the best performance with an R² score of 0.6875. In contrast, the Random Forest model has the lowest performance with an R² score of 0.5730.

## Contents

- **Models**: Implementations of CatBoost, Gradient Boosting, Extreme Gradient Boosting, and Random Forest Regressors.
- **Dataset**: The dataset used for training and evaluation, containing the aforementioned features and ZnO nanoparticle sizes.
- **Code**: Scripts for training, hyperparameter tuning, and evaluating the models.

## Results

- **Extreme Gradient Boosting**: Best model with an R² score of 0.6875.
- **Random Forest**: Lowest performing model with an R² score of 0.5730.

## Dataset

The dataset (Original_ZnO_dataset_2.csv) includes all necessary features to train and evaluate the models.

## Contributing

Feel free to submit issues or pull requests if you have any improvements or suggestions.

You can access the code and dataset here and start experimenting with the models to predict ZnO nanoparticle sizes.
