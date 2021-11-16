import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

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

    forest = RandomForestRegressor(
        n_jobs=-1,
        random_state=42,
        **params
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessing', preprocessor),
            ('scaling', scaler),
            ('model', forest)
        ]
    )

    return pipeline


if __name__ == '__main__':
    print('Importing data')
    df = pd.read_csv('Placement_Data_Full_Class.csv',
                     index_col='sl_no').reset_index(drop=True)
    df = df.dropna(subset=['salary']).reset_index(drop=True)

    print('Spliting data')
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42)

    numerical = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
    categorical = ['gender', 'ssc_b', 'hsc_b', 'hsc_s',
                   'degree_t', 'workex', 'specialisation']

    classification_target = ['status']
    regression_target = ['salary']

    X = df_full_train[numerical+categorical]
    y = df_full_train[regression_target]['salary']

    params = {'max_features': 'sqrt',
              'min_samples_split': 5,
              'n_estimators': 114}

    print('Creating pipeline')
    pipeline = create_new_pipeline(params)

    print('Training model')
    pipeline.fit(X, y)

    print('Saving model')
    with open('salary_model.pickle', 'wb') as f:
        pickle.dump((pipeline), f)
