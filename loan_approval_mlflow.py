
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Set project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# 2. Create CSV dataset if it does not exist
csv_file = "loan_approval.csv"

if not os.path.exists(csv_file):
    data = {
        "Applicant_Income": [
            50000, 30000, 80000, 25000, 60000,
            45000, 90000, 20000, 70000, 35000,
            55000, 28000, 85000, 40000, 65000,
            22000, 75000, 32000, 95000, 27000,
            58000, 42000, 88000, 18000, 62000,
            38000, 78000, 24000, 68000, 33000
        ],
        "Credit_Score": [
            750, 600, 800, 550, 720,
            680, 820, 500, 780, 620,
            740, 580, 810, 690, 760,
            520, 790, 610, 830, 570,
            730, 670, 805, 490, 750,
            640, 795, 530, 770, 590
        ],
        "Loan_Amount": [
            200000, 150000, 250000, 180000, 220000,
            190000, 300000, 120000, 240000, 160000,
            210000, 140000, 280000, 175000, 230000,
            110000, 260000, 155000, 320000, 130000,
            215000, 185000, 290000, 100000, 225000,
            165000, 270000, 115000, 245000, 145000
        ],
        "Loan_Status": [
            "Approved", "Rejected", "Approved", "Rejected",
            "Approved", "Approved", "Approved", "Rejected",
            "Approved", "Rejected", "Approved", "Rejected",
            "Approved", "Approved", "Approved", "Rejected",
            "Approved", "Rejected", "Approved", "Rejected",
            "Approved", "Approved", "Approved", "Rejected",
            "Approved", "Rejected", "Approved", "Rejected",
            "Approved", "Rejected"
        ]
    }

    pd.DataFrame(data).to_csv(csv_file, index=False)
    print("loan_approval.csv created successfully.")

# 3. Load dataset
df = pd.read_csv(csv_file)

print("\nLoan Approval Dataset:")
print(df.head())
print("\nDataset shape:", df.shape)

# 4. Separate input features and target
X = df[
    ["Applicant_Income", "Credit_Score", "Loan_Amount"]
]
y = df["Loan_Status"]

# 5. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Configure MLflow using SQLite
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Loan_Approval_Classification")

# 7. Train and compare two experiments
results = []
trained_models = {}

for depth in [2, 4]:

    with mlflow.start_run(
        run_name=f"max_depth_{depth}"
    ) as run:

        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

        # Train model
        model.fit(X_train, y_train)

        # Predict and calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters
        mlflow.log_param(
            "model_type", "DecisionTreeClassifier"
        )
        mlflow.log_param("max_depth", depth)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log metric
        mlflow.log_metric("accuracy", float(accuracy))

        # Log trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="loan_approval_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        results.append({
            "max_depth": depth,
            "accuracy": accuracy,
            "run_id": run.info.run_id
        })

        trained_models[depth] = model

        print(f"\nExperiment: max_depth={depth}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Run ID: {run.info.run_id}")

# 8. Compare experiments
comparison = pd.DataFrame(results)

print("\nExperiment Comparison:")
print(comparison[
    ["max_depth", "accuracy"]
].to_string(index=False))

# 9. Select the model with the higher test accuracy
best_depth = int(
    comparison.loc[
        comparison["accuracy"].idxmax(),
        "max_depth"
    ]
)

best_model = trained_models[best_depth]

print("\nBest max_depth:", best_depth)

# 10. Predict for a new loan applicant
new_applicant = pd.DataFrame([{
    "Applicant_Income": 65000,
    "Credit_Score": 750,
    "Loan_Amount": 200000
}])

prediction = best_model.predict(new_applicant)[0]

print("\nNew Applicant Details:")
print(new_applicant.to_string(index=False))
print("Loan Application Status:", prediction)

print("\nMLflow experiment:")
print("Loan_Approval_Classification")
