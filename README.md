# Predictive Modeling of Insurance Claim Legitimacy

This project implements a logistic regression model to predict whether an insurance claim is fraudulent or legitimate.  It uses a publicly available dataset of insurance claims that includes demographic information, policy details, claim history, claim amounts and other features.  The goal is to build a predictive model that can identify potentially fraudulent claims.

## Dataset

The dataset should be saved as `data/insurance_claims.csv`.  You can download a publicly available insurance fraud detection dataset (for example from Kaggle) and place it in the `data` directory.  The dataset typically includes columns such as claim amount, policyholder age, policy tenure, claim type, claim history and other categorical and numerical features.  Ensure that the target variable (whether the claim is fraudulent) is included in the dataset and encoded as `1` (fraudulent) or `0` (legitimate).

## Requirements

Install the required Python packages before running the script:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

Run the script to train the model and evaluate its performance:

```bash
python model.py
```

The script performs the following steps:

1. Loads the dataset from `data/insurance_claims.csv`.
2. Handles missing values, encodes categorical variables and scales numerical features.
3. Splits the data into training and testing sets.
4. Trains a logistic regression model.
5. Evaluates the model using accuracy, precision, recall and F1‑score and plots a confusion matrix.
6. Saves the confusion matrix plot in the `results` directory.

Modify the script if you wish to experiment with different models or preprocessing techniques.

## Results

If your dataset is similar to those used in the literature, a well‑tuned logistic regression model should achieve high accuracy (around 85–90%).  Use the evaluation metrics printed by the script to assess the quality of your model.

## Acknowledgements

This project is inspired by studies and datasets on insurance fraud detection and predictive analytics.
