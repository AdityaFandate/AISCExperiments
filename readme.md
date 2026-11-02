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

# Docker Web Application – Build and Run a Docker Container

## Aim

To create a Dockerfile for a sample web application, build a Docker image, run the container, and test the application in a web browser.

## Requirements

- Docker Desktop
- Visual Studio Code
- Web Browser (Google Chrome)
- Windows PowerShell

## Project Structure

```text
Dockerdemo/
│
├── Dockerfile
├── index.html
└── README.md
```

## 1. Create a Sample Web Application

Create a file named `index.html`.

### index.html

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Docker Web Application</title>
  </head>
  <body>
    <h1>Welcome to Docker!</h1>
    <h2>My First Docker Container</h2>
    <p>This web application is running inside a Docker container.</p>
  </body>
</html>
```

## 2. Create a Dockerfile

Create a file named `Dockerfile` without any extension.

### Dockerfile

```dockerfile
FROM nginx:latest

COPY index.html /usr/share/nginx/html/index.html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### Explanation

- `FROM nginx:latest` – Uses the official Nginx image.
- `COPY` – Copies the HTML file into the Nginx web directory.
- `EXPOSE 80` – Specifies the container port.
- `CMD` – Starts the Nginx web server.

## 3. Build the Docker Image

Open PowerShell inside the project directory and execute:

```bash
docker build -t my-web-app .
```

This command builds a Docker image named `my-web-app`.

## 4. Verify the Docker Image

```bash
docker images
```

This command displays all available Docker images.

## 5. Run the Docker Container

```bash
docker run -d -p 8080:80 --name my-container my-web-app
```

### Explanation

- `-d` – Runs the container in the background.
- `-p 8080:80` – Maps host port 8080 to container port 80.
- `--name` – Assigns a name to the container.
- `my-web-app` – Specifies the Docker image.

## 6. Check Running Containers

```bash
docker ps
```

This command displays all currently running containers.

## 7. Test the Application in a Web Browser

Open Google Chrome and enter:

```text
http://localhost:8080
```

Expected output:

```text
Welcome to Docker!

My First Docker Container

This web application is running
inside a Docker container.
```

## 8. Stop the Docker Container

```bash
docker stop my-container
```

To remove the container:

```bash
docker rm my-container
```

## Commands Summary

| Command                                                   | Description              |
| --------------------------------------------------------- | ------------------------ |
| `docker build -t my-web-app .`                            | Builds Docker image      |
| `docker images`                                           | Lists Docker images      |
| `docker run -d -p 8080:80 --name my-container my-web-app` | Runs container           |
| `docker ps`                                               | Lists running containers |
| `docker stop my-container`                                | Stops container          |
| `docker rm my-container`                                  | Removes container        |

## Result

Successfully created a Dockerfile, built a Docker image, and executed the sample web application inside a Docker container. The application was tested successfully using a web browser at `http://localhost:8080`.

## Conclusion

Docker helps package applications and their dependencies into containers. It makes applications easier to build, deploy, and run consistently across different environments.
