import pickle
import pandas as pd

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    with open('status_model.pickle', 'rb') as f:
        model = pickle.load(f)

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

    X = df_test[numerical+categorical]
    y = pd.get_dummies(df_test[classification_target])['status_Placed']

    score = model.score(X, y)

    print('Test set accuracy =', score)
