from sklearn import metrics


def validation(model_data):
    """
    getting the list from training data and storing it into y-test and y-predict ,
    @param model_data
    @return: accuracy
    """
    y_test = model_data[0]
    y_predict = model_data[1]
    accuracy = metrics.accuracy_score(y_test, y_predict)
    return accuracy

