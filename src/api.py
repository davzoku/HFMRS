import os
import logging

import pandas as pd
from huggingface_hub import HfApi


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO, 
    datefmt='%H:%M:%S'
    )


class API:
    """A class for retrieving information about Hugging Face models from the
    Hugging Face Hub API.

    :param data_path: Relative file path where the retrieved data will be saved 
    as a CSV file. Defaults to 'data/raw/data.csv'.
    :type data_path: str, optional
    :param return_df: Returns dataframe object if True, False otherwise. 
    Defaults to False.
    :type return_df: bool, optional
    """
    def __init__(self, 
                 data_path: str = 'data/raw/data.csv', 
                 return_df: bool = False):
        self.data_path = data_path
        self.return_df = return_df
        self.api = HfApi()
    
    def main(self):
        """Main function to automate API class methods. Calls 
        `_get_api_response`, `_extract_features`, `_to_dataframe`, and 
        `_export_to_csv` methods. Returns dataframe if `return_df` is True.
            
        :return: Pandas dataframe object
        :rtype: pandas.core.frame.DataFrame
        """
        res = self._get_api_response()
        self._extract_features(res)
        df = self._to_dataframe()
        self._export_to_csv(df)
        
        if self.return_df:  
            return df
    
    def _get_api_response(self):
        """Send a request to HuggingFace API and return response in a list.

        :return: List of models from Hugging Face API response
        :rtype: list
        """
        res = self.api.list_models(
            full=True, 
            cardData=True, 
            fetch_config=True, 
            sort=["downloads"], 
            direction=-1
            )
        
        logging.info('Retrieved API response')
        return list(res)
    
    def _extract_features(self, res: list):
        """Extract the required features from the API response.

        :param res: List of models from Hugging Face API response
        :type res: list
        """
        self.model_id = [
            self._catch(lambda: res[i].modelId, handle=lambda e: None) 
            for i in range(len(res))
            ]
        self.last_modified = [
            self._catch(lambda: res[i].lastModified, handle=lambda e: None) 
            for i in range(len(res))
            ]
        self.tags = [
            self._catch(lambda: res[i].tags, handle=lambda e: None) 
            for i in range(len(res))
            ]
        self.pipeline_tag = [
            self._catch(lambda: res[i].pipeline_tag, handle=lambda e: None) 
            for i in range(len(res))
            ]
        self.author = [
            self._catch(lambda: res[i].author, handle=lambda e: None) 
            for i in range(len(res))
            ]
        self.config_architectures = [
            self._catch(lambda: res[i].config['architectures'], 
                        handle=lambda e: None) for i in range(len(res))
            ]
        self.config_model_type = [
            self._catch(lambda: res[i].config['model_type'], 
                        handle=lambda e: None) for i in range(len(res))
            ]
        self.datasets = [
            self._catch(lambda: res[i].cardData['datasets'], 
                        handle=lambda e: None) for i in range(len(res))
            ]
        self.downloads = [
            self._catch(lambda: res[i].downloads, handle=lambda e: None) 
            for i in range(len(res))
            ]
        self.library_name = [
            self._catch(lambda: res[i].library_name, handle=lambda e: None) 
            for i in range(len(res))
            ]
        
        logging.info("Extracted features")

    def _to_dataframe(self):
        """Convert the extracted data from the Hugging Face API response to a 
        Pandas dataframe.
        
        :return: Pandas dataframe containing the extracted data
        :rtype: pandas.core.frame.DataFrame
        """
        df = pd.DataFrame({
            'modelId': self.model_id,
            'lastModified': self.last_modified,
            'tags': self.tags,
            'pipeline_tag': self.pipeline_tag,
            'author': self.author,
            'architectures': self.config_architectures,
            'model_type': self.config_model_type,
            'datasets': self.datasets,
            'downloads': self.downloads,
            'library_name': self.library_name
            })
        
        logging.info('Returning Pandas dataframe')
        return df
    
    def _export_to_csv(self, df: pd.DataFrame):
        """Export the extracted data to a CSV file.
        
        :param df: Pandas dataframe containing the extracted data
        :type df: pandas.core.frame.DataFrame
        """
        project_directory = os.path.dirname(os.path.abspath('.'))
        filepath = os.path.join(project_directory, self.data_path)
        data_directory = os.path.dirname(filepath)
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        df.to_csv(filepath, index=False)
        
        logging.info(f"Exported to csv file - {filepath}")
    
    @staticmethod
    def _catch(func: callable, *args, handle: callable = lambda e: e, **kwargs):
        """Call a function and catch any exceptions during the function call.

        :param func: Function to call
        :type func: callable
        :param handle: Function that takes in an exception as an argument and 
        return a default value. Defaults to lambda e: e.
        :type handle: callable, optional
        :return: Result of calling the function, or the value returned by handle 
        if an exception occurred.
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return handle(e)