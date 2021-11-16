import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import pickle

import warnings
warnings.filterwarnings("ignore")


def create_new_pipeline(params):
    numerical_transformer = SimpleImputer(strategy='mean')

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoding', OneHotEncoder(drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical),
            ('categorical', categorical_transformer, categorical)
        ])

    scaler = StandardScaler()

    logreg = XGBClassifier(
        n_jobs=-1,
        random_state=42,
        **params
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessing', preprocessor),
            ('scaling', scaler),
            ('model', logreg)
        ]
    )

    return pipeline


if __name__ == '__main__':
    print('Importing data')
    df = pd.read_csv('Placement_Data_Full_Class.csv',
                     index_col='sl_no').reset_index(drop=True)

    print('Spliting data')
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42)

    numerical = ['hsc_p', 'degree_p', 'ssc_p']
    categorical = ['gender', 'ssc_b', 'hsc_b', 'hsc_s',
                   'degree_t', 'workex', 'specialisation']

    classification_target = ['status']
    regression_target = ['salary']

    X = df_full_train[numerical+categorical]
    y = pd.get_dummies(df_full_train[classification_target])['status_Placed']

    params = {'learning_rate': 0.5272631578947369,
              'max_depth': 6,
              'n_estimators': 10,
              'reg_alpha': 0.1,
              'reg_lambda': 1.0}

    print('Creating pipeline')
    pipeline = create_new_pipeline(params)

    print('Training model')
    pipeline.fit(X, y)

    print('Saving model')
    with open('status_model.pickle', 'wb') as f:
        pickle.dump((pipeline), f)
