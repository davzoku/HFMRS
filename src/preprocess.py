import logging
import os
import re
import string

import numpy as np
import pandas as pd


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO, 
    datefmt='%H:%M:%S'
    )


class Preprocess:
    """Class for preprocessing raw CSV data for model ingestion.

    :param input_path: Relative file path of input raw CSV file. 
    Defaults to './data/raw/data.csv'.
    :type input_path: str, optional
    :param output_path: Relative file path where the retrieved data will be 
    saved as a CSV file. Defaults to 'data/raw/data.csv'.
    :type output_path: str, optional
    :param limit: Setting to limit number of samples. Defaults to None.
    :type limit: int, optional
    :param return_df: _Returns dataframe object if True, False otherwise. 
    Defaults to False.
    :type return_df: bool, optional
    """
    def __init__(self,
                 input_path: str = './data/raw/data.csv',
                 output_path: str = './data/processed/data.csv',
                 limit: int = None,
                 return_df: bool = False):
        self.df = pd.read_csv(input_path)
        self.output_path = output_path
        self.limit = limit
        self.return_df = return_df
    
    def main(self):
        """Main function to automate Preprocess class methods. Calls 
        `_process_list_features`, `_drop_columns`, `_limit_dataset`, 
        `_create_soup` and `_export_to_csv` methods. 
        
        Returns dataframe if `return_df` is True.
            
        :return: Pandas dataframe object
        :rtype: pandas.core.frame.DataFrame
        """
        self._process_list_features()
        self._drop_columns()
        self._limit_dataset()
        self.df['soup'] = self.df.apply(self._create_soup, axis=1)
        self._export_to_csv()
        
        if self.return_df:  
            return self.df
    
    def _process_list_features(self, features: list = \
                              ['tags', 'architectures']):
        """Process features with list values to string values and impute 
        exceptions values with NaN values.

        :param features: Features with list values. 
        Defaults to \['tags', 'architectures'].
        :type features: list, optional
        """
        for feat in features:
            self.df[feat] = self.df[feat].apply(str)
            self.df[feat] = self.df[feat].apply(
                lambda x: re.sub('[%s]' % re.escape(string.punctuation), "", x)
                )
            self.df[feat] = self.df[feat].apply(lambda x: x.lower())
            self.df[feat].replace('[]', np.nan, inplace=True)
        
        logging.info("Preprocessed features with list values")

    
    def _drop_columns(self, columns: list = ['lastModified', 'datasets']):
        """Drop column(s) in Pandas DataFrame.

        :param columns: List of column(s) to drop. Defaults to ['lastModified'].
        :type columns: list, optional
        """
        self.df.drop(labels=columns, axis=1, inplace=True)
        logging.info(f"Dropped columns - {columns}")

    def _limit_dataset(self):
        """Limit number of samples/rows in DataFrame based on user input.
        """
        self.df = self.df[:self.limit]
        logging.info(f"Limit dataset to {self.limit} rows")

    def _create_soup(self, x: pd.Series):
        """Feature engineer soup feature by concatenating all the string values.
        
        :param x: Pandas Series representing a row in the DataFrame.
        :type x: str
        :return: A string representing the soup features for the given row.
        :rtype: str
        """
        string_features = \
            self.df.drop(['modelId', 'downloads'], axis=1).columns
        soup = \
            ' '.join(str(val) for val in x[string_features].values.flatten())
        
        return soup
    
    def _export_to_csv(self):
        """Export the extracted data to a CSV file.
        """
        project_directory = os.path.dirname(os.path.abspath('.'))
        filepath = os.path.join(project_directory, self.output_path)
        data_directory = os.path.dirname(filepath)
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        self.df.to_csv(filepath, index=False)
        
        logging.info(f"Exported to csv file - {filepath}")
