import unittest
import logging

import pandas as pd

from src.utils.constant import file_path


class Testing_Svm_Classification:
    def __init__(self):
        """
        getting data from preprocessing function and calculating the accuracy of svm
         @param credit-card-dataframe
        @type credit-card-dataframe
        """
        read_file = pd.read_csv(file_path)
        self.credit_card_dataframe = pd.DataFrame(read_file)

    def test_pre_processing(self):
        """
        getting the dataframe and pre processing it
        @param dataframe
        @return dataframe
        @rtype list
        """
        read_file = pd.read_csv(file_path)
        data_frame = pd.DataFrame(read_file)

        if isinstance(data_frame, pd.DataFrame):
            logging.info("This is credit card Dataframe")
        else:
            logging.info("This is not Dataframe")

    def test_target_feature(self):
        """
        getting the dataframe and dividing it into X_feature and target
        @param self.dataset
        @return: dataframe
        """

        X_feature = self.credit_card_dataframe.drop("defaulter", axis=1)
        target = self.credit_card_dataframe["defaulter"]

        if len(X_feature) > 0 and ("defaulter" not in X_feature.columns):
            logging.info("Features are divided correctly")
        else:
            logging.info("Features are not divided correctly")


class Test_class(unittest.TestCase):
    def test_for_dataframe(self):
        with self.assertLogs() as captured:
            check = Testing_Svm_Classification()
            check.test_pre_processing()
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "This is credit card Dataframe")

    def test_for_dividing_features(self):
        with self.assertLogs() as captured:
            check = Testing_Svm_Classification()
            check.test_target_feature()
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "Features are divided correctly")


if __name__ == '__main__':
    unittest.main()
