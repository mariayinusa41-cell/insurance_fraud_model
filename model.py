import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and encoding categorical variables.

    Parameters:
        df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df = df.copy()
    # Drop rows with missing values
    df = df.dropna()
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded


def train_and_evaluate(df: pd.DataFrame, target_column: str):
    """
    Train a logistic regression model and evaluate its performance.

    Parameters:
        df (pd.DataFrame): The DataFrame containing features and target.
        target_column (str): The name of the target column.

    Returns:
        model: The trained logistic regression model.
        float: The accuracy of the model on the test data.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Create and save a confusion matrix plot
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", "confusion_matrix.png"))
    plt.close()

    return model, accuracy


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and evaluate a logistic regression model for insurance claim legitimacy."
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the CSV file containing the dataset."
    )
    parser.add_argument(
        "--target", type=str, required=True, help="Name of the target column."
    )
    args = parser.parse_args()

    data_path = args.data
    target_column = args.target

    df = load_data(data_path)
    df_processed = preprocess(df)
    model, acc = train_and_evaluate(df_processed, target_column)
    print(f"Accuracy: {acc:.2f}")


if __name__ == "__main__":
    main()
