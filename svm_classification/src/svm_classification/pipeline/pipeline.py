from src.svm_classification.model_training.model_training import training_model
from src.svm_classification.model_validation.model_validation import validation
from src.svm_classification.pre_processing.processing import pre_processing
from src.utils.constant import file_path


class Svm_Classification:
    def __init__(self):
        """
        setting the reference of dataframe and pipeline function
         @param dataframe
        @type dataframe
        """
        self.dataset = file_path
        self.pipeline()

    def pipeline(self):
        """
        getting the dataframe and calling all the steps in building the model
        @return: accuracy
        """
        processed_dataframe = pre_processing(self.dataset)
        training = training_model(processed_dataframe)
        accuracy = validation(training)
        print("Accuracy: ",accuracy)



