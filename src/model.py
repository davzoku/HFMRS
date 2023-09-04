import logging
import os
import json

import pandas as pd
import numpy as np
import torch

from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sentence_transformers import SentenceTransformer

from gnn import *

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)


class Model:
    def __init__(
        self,
        data_path: str = "./data/processed/data.csv",
        embed_path: str = "./data/embed.npy",
        model_path: str = "./model/gnn.pt",
        mapping_path: str = "./model/model_mapping.json",
        downloads_path: str = "./model/downloads_mapping.json",
        input_feat: str = "modelId",
    ):
        """
        Initialize a Model instance for front-end inference.

        :param data_path: The path to the processed data file to use for inference. This should be a CSV file with a column corresponding to the input feature specified by `input_feat`.
        :type data_path: str
        :param embed_path: The path to a NumPy file containing pre-trained embeddings for the data. The embeddings should be in the form of a matrix with rows corresponding to the data items and columns corresponding to the embedding dimensions.
        :type embed_path: str
        :param model_path: The path to a PyTorch model file to use for inference. This is not currently used and is a work-in-progress (WIP).
        :type model_path: str
        :param input_feat: The name of the column in the data file corresponding to the input feature to use for inference.
        :type input_feat: str

        :returns: None

        :raises IOError: If the data or embedding file cannot be found or loaded.

        """

        self.df = pd.read_csv(data_path)
        self.input_feat = input_feat
        self.mapping_path = mapping_path
        self.downloads_path = downloads_path

        # Extract embeddings using torch pretrained embedding
        if os.path.exists(embed_path):
            self.embeddings = self._load_embed(embed_path)
        else:
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = embedder.encode(self.df["soup"], convert_to_tensor=False)
            self._save_embed()

        logging.info("Loaded Pretrained Embeddings")

        self.gnn = self._load_model(
            model_path,
            mapping_path=self.mapping_path,
            downloads_path=self.downloads_path,
        )

        logging.info("Loaded GNN model")

        # Construct a reverse map of indices
        self.indices = pd.Series(
            self.df.index, index=self.df[self.input_feat]
        ).drop_duplicates()

        logging.info("Initialized Index Map")
        self.cosine_sim = self._cosine_similarity()
        logging.info("Initialized Cosine Similarity Scores")

        self.binary_matrix = np.where(self.embeddings > 0, 1, 0)
        self.jaccard_sim = self._jaccard_similarity(self.binary_matrix)
        logging.info("Initialized Jaccard Similarity Scores")

    def get_recommendation(self, input: str, metric: str = "cosine"):
        """Get top 30 recommendations

        :param input: The model ID to retrieve recommendations for.
        :type input: str
        :param metric: The metric to use for similarity calculations. Currently supports "cosine" and "jaccard", defaults to "cosine".
        :type metric: str, optional
        :return: A dataframe containing the top 30 recommendations for the specified model ID.
        :rtype: pandas.core.frame.DataFrame
        :raises ValueError: Raised when the specified metric is not supported.
        """
        if input in self.df["modelId"].values:
            logging.info(f"Find model {input}")
            idx = self.indices[input]

            if metric.lower() == "cosine":
                # Get the pairwise similarity scores
                sim_scores = list(enumerate(self.cosine_sim[idx]))

                # Sort based on the similarity scores
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

                # Get top 30 scores
                sim_scores = sim_scores[1:30]

                # Get indices
                return_indices = [i[0] for i in sim_scores]
                small_df = self.df.iloc[return_indices].copy()
                small_df["score"] = [i[1] for i in sim_scores]

            if metric.lower() == "jaccard":
                similar_indices = self.jaccard_sim[idx].argsort()[::-1][1:31]
                small_df = self.df.iloc[similar_indices].copy()
                small_df["score"] = self.jaccard_sim[idx][similar_indices]

            if metric.lower() == "gnn":
                small_df = self._gnn_inference(input)

            else:
                print(f"{metric} not supported")

        else:
            print(f'Error: product_name "{input}" not found in dataframe.')

        # Return the top 30 for frontend
        return small_df

    def _load_embed(self, path):
        """
        Load an embedding matrix from a NumPy .npy file at the given path.

        Parameters:
            path (str): The path to the .npy file containing the embedding matrix.

        Returns:
            numpy.ndarray: The loaded embedding matrix.

        Raises:
            FileNotFoundError: If the file at the given path does not exist.
        """
        return np.load(path, allow_pickle=True)

    def _save_embed(self, path="./data/embed"):
        np.save(path, self.embeddings)

    def _load_model(
        self,
        path,
        mapping_path: str = "./model/model_mapping.json",
        downloads_path: str = "./model/downloads_mapping.json",
    ):
        """
        Load a PyTorch model from a file at the given path and set it to evaluation mode.

        Parameters:
            path (str): The path to the file containing the saved PyTorch model.

        Returns:
            torch.nn.Module: The loaded PyTorch model.

        Raises:
            FileNotFoundError: If the file at the given path does not exist.
        """

        with open(mapping_path, "r") as f:
            model_mapping = json.load(f)
            model_mapping = {int(key): value for key, value in model_mapping.items()}
        with open(downloads_path, "r") as f:
            downloads_mapping = json.load(f)
            downloads_mapping = {
                int(key): value for key, value in downloads_mapping.items()
            }

        model = RecommenderSystem(
            model_mapping,
            downloads_mapping,
            in_channels=2688,
            hidden_channels=32,
            out_channels=1,
        )

        model.load_state_dict(torch.load(path))
        model.eval()

        return model

    def _cosine_similarity(self):
        """Calculates Cosine Similarity Scores.

        :return: Cosine Similarity Scores
        :rtype: ndarray
        """
        return cosine_similarity(self.embeddings)

    def _jaccard_similarity(self, binary_matrix):
        """
        Calculate the pairwise Jaccard similarity between rows in a binary matrix.

        Parameters:
            binary_matrix (numpy.ndarray): The binary matrix for which to compute the pairwise Jaccard similarity.

        Returns:
            numpy.ndarray: A square matrix of pairwise Jaccard similarities, where each entry (i, j) represents the Jaccard similarity between rows i and j.

        Notes:
            The Jaccard similarity between two sets is defined as the size of their intersection divided by the size of their union. Here, the binary matrix is treated as a collection of sets, where each row represents a set of items, and the Jaccard similarity is computed between pairs of sets (i.e., rows).

        References:
            - https://en.wikipedia.org/wiki/Jaccard_index
            - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
        """
        return 1 - pairwise_distances(binary_matrix, metric="jaccard")

    def _gnn_inference(self, input):
        """
        Use the graph neural network (GNN) to make predictions for a given input.

        Parameters:
            input (numpy.ndarray): An input feature vector or matrix.

        Returns:
            numpy.ndarray: The predicted output vector or matrix, computed using the trained GNN.

        Notes:
            The GNN is trained to map input feature vectors to output vectors. This method applies the trained GNN to the input, returning the corresponding predicted output.

        References:
            - https://en.wikipedia.org/wiki/Graph_neural_network
        """
        return self.gnn.predict(input, 30)
