from sklearn.metrics import f1_score

from models.train_model import train_model
from models.predict_model import predict_model
from features.build_features import build_features

if __name__ == "__main__":
    data_file_path = "../data/raw/ACME-HappinessSurvey2020.csv"
    X_train, X_test, y_train, y_test = build_features(data_file_path)
    model = train_model(X_train, y_train)
    y_pred = predict_model(model, X_test)
    score = f1_score(y_test, y_pred)
    print(f"F1 Score: {score:.3f}")
