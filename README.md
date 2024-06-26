# Triglyceride Level Prediction Project

## Overview
This project focuses on developing a machine learning model to predict triglyceride levels based on various health and lifestyle factors. Triglyceride levels are important indicators of heart health, and predicting them accurately can help in assessing an individual's risk of cardiovascular diseases.

## Dataset
The dataset used for this project contains information about candidates including their gender, age, height, weight, smoking and drinking habits, glucose level, creatinine level, and other health parameters. The dataset is divided into training and test sets for model development and evaluation.

## Code Structure
- **main.py**: Contains the main code for data preprocessing, model training, evaluation, and prediction on the test set.
- **train.csv**: Training dataset containing candidate information and triglyceride levels.
- **test.csv**: Test dataset without triglyceride levels for prediction.
- **sample_submission.csv**: Sample submission file format for predictions.

## Workflow
1. **Data Loading and Preprocessing**: Load the training and test datasets using pandas. Combine and preprocess the data, handling missing values and encoding categorical variables.
2. **Model Training**: Split the data into training and validation sets. Train an XGBoost regressor model on the training data.
3. **Model Evaluation**: Evaluate the model's performance using Mean Absolute Error (MAE) on the validation set.
4. **Prediction on Test Data**: Make predictions on the test data using the trained model.
5. **Submission**: Generate a submission file with candidate IDs and predicted triglyceride levels for evaluation and sharing.

## Usage
1. Ensure Python and required libraries (pandas, scikit-learn, xgboost) are installed.
2. Run `python main.py` to execute the code.
3. Check the output for Mean Absolute Error (MAE) on the validation set and the generated submission file.

## Conclusion
This project demonstrates how machine learning techniques can be applied to predict health-related parameters like triglyceride levels based on individual characteristics. The model's predictions can aid in assessing cardiovascular risks and promoting healthier lifestyles.

