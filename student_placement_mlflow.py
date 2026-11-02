
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Set the tracking directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. Load the dataset
df = pd.read_csv("student_placement.csv")

print("Dataset:")
print(df.head())
print("\nDataset shape:", df.shape)

# 3. Separate features and target
X = df[[
    "CGPA",
    "Aptitude_Score",
    "Technical_Skills_Score"
]]
y = df["Placement_Status"]

# 4. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Create MLflow experiment
mlflow.set_experiment("Student_Placement_Classification")

# 6. Compare two Decision Tree experiments
results = []
trained_models = {}

for depth in [2, 4]:

    with mlflow.start_run(
        run_name=f"max_depth_{depth}"
    ) as run:

        # Create and train model
        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

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

        # Log accuracy
        mlflow.log_metric("accuracy", float(accuracy))

        # Log trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="student_placement_model",
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

        print(f"\nMax depth: {depth}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Run ID: {run.info.run_id}")

# 7. Compare experiments
comparison = pd.DataFrame(results)

print("\nExperiment Comparison:")
print(comparison[
    ["max_depth", "accuracy"]
].to_string(index=False))

# 8. Select the better model for demonstration
best_depth = int(
    comparison.loc[
        comparison["accuracy"].idxmax(),
        "max_depth"
    ]
)

best_model = trained_models[best_depth]

print("\nBest max_depth:", best_depth)

# 9. Predict for a new student
new_student = pd.DataFrame([{
    "CGPA": 8.2,
    "Aptitude_Score": 80,
    "Technical_Skills_Score": 84
}])

prediction = best_model.predict(new_student)[0]

print("\nNew Student:")
print(new_student)
print("Predicted Placement Status:", prediction)

print("\nMLflow experiment:")
print("Student_Placement_Classification")
