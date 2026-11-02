# MLflow Classification Experiments

This project contains three machine learning classification models using a Decision Tree Classifier and MLflow.

1. **Student Placement Prediction** — Placed or Not Placed
2. **Loan Approval Prediction** — Approved or Rejected
3. **Employee Attrition Prediction** — Stay or Leave

Each program trains two models with different `max_depth` values, calculates accuracy, and logs parameters, metrics, and trained models using MLflow.

## 1. Project Structure

Keep the following files in your `AISCExperiments` folder:

```text
AISCExperiments/
├── student_placement_mlflow.py
├── student_placement.csv
├── loan_approval_mlflow.py
├── loan_approval.csv
├── employee_attrition_mlflow.py
├── employee_attrition.csv
└── mlflow.db
```

The loan and employee scripts create their CSV datasets automatically if the files do not exist. Ensure `student_placement.csv` is also present.

## 2. Install Required Libraries

Open PowerShell in the project folder and run:

```powershell
python -m pip install pandas scikit-learn mlflow
```

## 3. Configure MLflow

In all three Python scripts, use the same tracking URI before calling `mlflow.set_experiment()`:

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

This stores all experiments in one SQLite database.

## 4. Execute the Three Programs

Run each command separately from the project folder.

### A. Student Placement

```powershell
python student_placement_mlflow.py
```

Experiment name: `Student_Placement_Classification`

Features:

- CGPA
- Aptitude Score
- Technical Skills Score

Target: `Placed` or `Not Placed`

### B. Loan Approval

```powershell
python loan_approval_mlflow.py
```

Experiment name: `Loan_Approval_Classification`

Features:

- Applicant Income
- Credit Score
- Loan Amount

Target: `Approved` or `Rejected`

### C. Employee Attrition

```powershell
python employee_attrition_mlflow.py
```

Experiment name: `Employee_Attrition_Classification`

Features:

- Years of Experience
- Monthly Salary
- Job Satisfaction Score

Target: `Stay` or `Leave`

Each script compares Decision Tree models with `max_depth=2` and `max_depth=4`.

## 5. Start the MLflow Dashboard

After executing the Python scripts, run:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Keep the terminal running and open this URL in your browser:

http://127.0.0.1:5000

## 6. View and Compare Experiments

In the MLflow dashboard, open the Experiments page. You should find:

- `Student_Placement_Classification`
- `Loan_Approval_Classification`
- `Employee_Attrition_Classification`

Open each experiment to view its two runs:

- `max_depth_2`
- `max_depth_4`

Compare the following logged information:

| Parameter      | Description                                        |
| -------------- | -------------------------------------------------- |
| `max_depth`    | Maximum depth of the Decision Tree                 |
| `test_size`    | Proportion of data used for testing                |
| `random_state` | Fixed seed for reproducible splitting and training |
| `accuracy`     | Proportion of correct test predictions             |
| Model artifact | Saved trained Decision Tree model                  |

The accuracy values are calculated when the scripts run.

## 7. Execution Order

1. Install the required libraries.
2. Make sure all three Python scripts and the student CSV are in the project folder.
3. Run the student placement script.
4. Run the loan approval script.
5. Run the employee attrition script.
6. Start the MLflow dashboard.
7. Open the dashboard and compare the experiments.

Use a second PowerShell terminal if you want to run more scripts while the dashboard is running.

## 8. Troubleshooting

**If `mlflow` is not recognized:**

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**If the CSV file is not found:**

```powershell
dir
```

Check that the filename and folder are correct.

**If the filesystem tracking backend error occurs:**

Ensure all scripts use:

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

Start the dashboard with the same SQLite URI.

**If model logging fails because of `skops_trusted_types`:**

Check your installed MLflow version:

```powershell
python -m mlflow --version
```

Use a model-logging argument supported by your installed version.

## 9. Result

Three classification models are developed and evaluated using Decision Trees. MLflow records their parameters, accuracy metrics, and trained models, enabling comparisons between the two `max_depth` settings for each experiment.

**Note:** The datasets are illustrative educational examples and should not be used for actual lending or employment decisions.

GitHub 4 5 and 6
https://chatgpt.com/share/6ae8b1f8-4700-83e8-81bf-06a18c5f41b9
