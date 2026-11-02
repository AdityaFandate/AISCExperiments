
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
csv_file = "employee_attrition.csv"

if not os.path.exists(csv_file):
    data = {
        "Years_of_Experience": [
            1, 3, 8, 2, 10, 5, 12, 4, 7, 1,
            9, 6, 15, 3, 11, 2, 8, 5, 14, 4,
            10, 1, 6, 13, 3, 9, 2, 12, 7, 5
        ],
        "Monthly_Salary": [
            25000, 35000, 65000, 28000, 80000,
            45000, 95000, 38000, 60000, 22000,
            72000, 50000, 110000, 32000, 85000,
            26000, 62000, 48000, 105000, 40000,
            78000, 24000, 52000, 100000, 30000,
            70000, 27000, 90000, 58000, 46000
        ],
        "Job_Satisfaction_Score": [
            2, 3, 5, 2, 5, 4, 5, 3, 4, 1,
            5, 3, 5, 2, 4, 1, 4, 3, 5, 2,
            4, 1, 3, 5, 2, 4, 2, 5, 3, 4
        ],
        "Attrition_Status": [
            "Leave", "Stay", "Stay", "Leave", "Stay",
            "Stay", "Stay", "Leave", "Stay", "Leave",
            "Stay", "Stay", "Stay", "Leave", "Stay",
            "Leave", "Stay", "Stay", "Stay", "Leave",
            "Stay", "Leave", "Stay", "Stay", "Leave",
            "Stay", "Leave", "Stay", "Stay", "Stay"
        ]
    }

    pd.DataFrame(data).to_csv(csv_file, index=False)
    print("employee_attrition.csv created successfully.")

# 3. Load dataset
df = pd.read_csv(csv_file)

print("\nEmployee Attrition Dataset:")
print(df.head())
print("\nDataset shape:", df.shape)

# 4. Separate input features and target
X = df[
    [
        "Years_of_Experience",
        "Monthly_Salary",
        "Job_Satisfaction_Score"
    ]
]
y = df["Attrition_Status"]

# 5. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Configure MLflow with SQLite
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Employee_Attrition_Classification")

# 7. Train and compare two experiments
results = []
trained_models = {}

for depth in [2, 4]:

    with mlflow.start_run(
        run_name=f"max_depth_{depth}"
    ) as run:

        # Create and train Decision Tree
        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict and calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log model parameters
        mlflow.log_param(
            "model_type", "DecisionTreeClassifier"
        )
        mlflow.log_param("max_depth", depth)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log accuracy metric
        mlflow.log_metric("accuracy", float(accuracy))

        # Log trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="employee_attrition_model",
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

# 8. Compare both experiments
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

# 10. Predict for a new employee
new_employee = pd.DataFrame([{
    "Years_of_Experience": 6,
    "Monthly_Salary": 55000,
    "Job_Satisfaction_Score": 4
}])

prediction = best_model.predict(new_employee)[0]

print("\nNew Employee Details:")
print(new_employee.to_string(index=False))
print("Predicted Attrition Status:", prediction)

print("\nMLflow experiment:")
print("Employee_Attrition_Classification")
