import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    with open('salary_model.pickle', 'rb') as f:
        model = pickle.load(f)

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

    X = df_test[numerical+categorical]
    y = df_test[regression_target]['salary']

    score = mean_squared_error(model.predict(X), y, squared=False)

    print('Test set RMSE =', score)
