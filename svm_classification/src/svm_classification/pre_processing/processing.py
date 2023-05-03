import pandas as pd
from pandas import DataFrame

from sklearn.model_selection import train_test_split


def pre_processing(data_frame):
    """
    getting the dataframe and pre processing it
    @param data_frame
    @return dataframe
    """
    read_file = pd.read_csv(data_frame)
    data_frame = pd.DataFrame(read_file)
    processed_data = target_feature(data_frame)
    return processed_data


def target_feature(credit_card_dataframe):
    """
    getting the dataframe and dividing it into X-feature and target.
    @param credit_card_dataframe
    @return: dataframe
    """
    X_feature = credit_card_dataframe.drop("defaulter", axis=1)
    target = credit_card_dataframe["defaulter"]
    pre_processed_data = splitting_dataset(X_feature, target)
    return pre_processed_data


def splitting_dataset(X_feature, target):
    """
    getting x-feature and target from target-feature function and spiltting them into training and testing sets
    using train-test-split
    @param X_feature , target
    @return : list[X_train, X_test, y_train, y_test]
    """
    X_train, X_test, y_train, y_test = train_test_split(X_feature, target, test_size=0.20, random_state=50)
    splitted_data = [X_train, X_test, y_train, y_test]
    return splitted_data
