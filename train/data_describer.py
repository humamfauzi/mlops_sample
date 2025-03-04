from abc import ABC, abstractmethod

class TabularDataDescriber(ABC):
    @abstractmethod
    def describe_all_numerical_data(self):
        '''
        Provides a statistical summary of the raw data and numerical.
        This method returns a statistical summary of the raw data, including
        measures such as mean, standard deviation, min, and max values for each
        column in the dataset. The raw data must be loaded before calling this method.
        Raises:
            ValueError: If the raw data has not been loaded.
        Returns:
            pandas.DataFrame.describe: A DataFrame containing the statistical summary of the raw data.
        '''
        pass

    @abstractmethod
    def describe_all_categorical_data(self):
        '''
        Provides a summary of the raw categorical data.
        This method returns a summary of the raw categorical data, including
        measures such as unique values, frequency of the most common value, etc.
        The raw data must be loaded before calling this method.
        Raises:
            ValueError: If the raw data has not been loaded.
        Returns:
            pandas.DataFrame.describe: A DataFrame containing the summary of the raw categorical data.
        '''
        pass

class RegularDescriber(TabularDataDescriber):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def describe_all_numerical_data(self):
        return self.data_loader.describe_all_numerical_data()

    def describe_all_categorical_data(self):
        return self.data_loader.describe_all_categorical_data()