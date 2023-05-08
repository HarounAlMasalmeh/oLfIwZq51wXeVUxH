from sklearn.neighbors import KNeighborsClassifier


def train_model(X_train, y_train):
    KNN = KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='auto', leaf_size=2, p=1)
    KNN.fit(X_train, y_train)
    return KNN
