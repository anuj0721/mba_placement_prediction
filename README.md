# MBA Placement Prediction

## Description

The [dataset](https://www.kaggle.com/benroshan/factors-affecting-campus-placement) consists of placement data of students in an MBA university. It includes secondary and higher secondary school percentages and specialization. It also includes degree information, work experience, and salary offered to the placed students.

The goal of this project is to create an ML model to predict which students will get placed and the salary of the placed students.

The files starting with ‘status*’ are for the classification problem where the model predicts whether a student will get placed or not. The files starting with ‘salary*’ are for the regression problem where the model predicts the salary offered to the placed students.

For the status prediction problem, I have trained three models: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and [XGBClassifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier).

For the salary prediction problem, I have trained four models: [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html), [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) and [XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor).

I have created [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) for each of these models, such that the data transformation and model training/predictions steps are assembled together.

For both the problems, I have done separate EDA and feature selection.

## Tech Stack and concepts used

- Python
- Scikit-learn
- Machine Learning Pipeline
- Docker
- Streamlit

## Setup

- Clone the project repo and open it.

### Virtual Environment

- Create a virtual environment for the project using

  ```bash
  pipenv shell
  ```

- Install required packages using

  ```bash
  pipenv install
  ```

### Docker Container

- Build the docker image using

  ```bash
  docker build -t mba_placement .
  ```

- Run the docker container using

  ```bash
  docker run -p 5000:5000 mba_placement
  ```

- Open the URL http://localhost:5000/ to run and test the app.

### Deploying to Cloud

- Open the [Deploy an app](https://share.streamlit.io/deploy) page of Streamlit.
- Enter the GitHub repository details in which the streamlit_app.py file and model binaries are stored.
- Click on Deploy button.
- Open the URL https://share.streamlit.io/anuj0721/mba_placement_prediction/master to run and test the app.

## Status Prediction Results

| Model                  | Validation Set Accuracy | Training+Validation Set Accuracy |
| ---------------------- | ----------------------- | -------------------------------- |
| LogisticRegression     | 95.35 %                 | 89.53 %                          |
| RandomForestClassifier | 93.02 %                 | 96.51 %                          |
| XGBClassifier          | 97.67 %                 | 98.84 %                          |

Selected Model (XGBClassifier) Test Set Accuracy = 83.72 %

## Salary Prediction Results

| Model                 | Validation Set RMSE | Training+Validation Set RMSE |
| --------------------- | ------------------- | ---------------------------- |
| LinearRegression      | 72827.10            | 84563.22                     |
| ElasticNet            | 58410.36            | 89727.80                     |
| RandomForestRegressor | 58509.22            | 51349.49                     |
| XGBRegressor          | 60382.35            | 53142.14                     |

Selected Model (RandomForestRegressor) Test Set RMSE = 92649.18
