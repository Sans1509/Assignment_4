from sklearn import svm


def training_model(data_frame):
    """
        getting the list from preprocessing function and dividing it into 4 train and test data and calculating
        y-predict
         @param : List
        @return: List
        """

    X_train = data_frame[0]
    X_test = data_frame[1]
    y_train = data_frame[2]
    y_test = data_frame[3]

    svm_classifier = svm.SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)
    y_predict = svm_classifier.predict(X_test)
    data = [y_test, y_predict]

    return data
