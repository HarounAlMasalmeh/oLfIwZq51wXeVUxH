import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_features(file_path):
    data = pd.read_csv(file_path)
    data.drop_duplicates(keep='first', inplace=True)

    X = data.drop(['Y', 'X2', 'X4'], axis=1)
    y = data['Y']

    X_train_init, X_test_init, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=36, stratify=y)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    X_train = numerical_transformer.fit_transform(X_train_init)
    X_test = numerical_transformer.transform(X_test_init)

    return X_train, X_test, y_train, y_test
