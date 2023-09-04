import json
import os
import logging

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from tqdm import tqdm


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S'
)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # x = F.sigmoid(x)
        x = F.softmax(x, dim=0)
        return x


class RecommenderSystem(torch.nn.Module):
    def __init__(self, 
                 modelId_mapping, 
                 downloads_mapping,
                 in_channels, 
                 hidden_channels, 
                 out_channels):
        super(RecommenderSystem, self).__init__()
        self.modelId_mapping = modelId_mapping
        self.downloads_mapping = downloads_mapping
        self.inverted_mapping = {v: k for k, v in modelId_mapping.items()}
        self.nodes = \
            torch.tensor(list(modelId_mapping.keys()), dtype=torch.long)
        self.num_nodes = len(modelId_mapping)
        
        self.embedding = torch.nn.Embedding(self.num_nodes, in_channels)
        self.gnn = GNN(in_channels, hidden_channels, out_channels)
        self.device = \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, nodes, edge_index, edge_weight):
        x = self.embedding(nodes)
        x = self.gnn(x, edge_index, edge_weight)
        
        return x
    
    def train_model(self,
                    data, 
                    modelId_mapping,
                    optimizer, 
                    criterion, 
                    num_epochs):  
        self.train()
        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()
            out = self(
                torch.tensor(list(modelId_mapping.keys()), dtype=torch.long),
                data.edge_index, 
                data.edge_weight
                )
            loss = criterion(
                out[data.train_mask], 
                data.y[data.train_mask].unsqueeze(1).float()
                )
            loss.backward()
            optimizer.step()
            
            
    def evaluate_model(self, 
                       data, 
                       modelId_mapping, 
                       criterion):
        self.eval()
        with torch.no_grad():
            out = self(
                torch.tensor(list(modelId_mapping.keys()), dtype=torch.long), 
                data.edge_index, 
                data.edge_weight
                )
            test_loss = criterion(
                out[data.test_mask], 
                data.y[data.test_mask].unsqueeze(1).float()
                )
    
    def predict(self, model_id, top_k): #data, model_id, modelId_mapping, top_k):
        # inverted_mapping = {v: k for k, v in modelId_mapping.items()}
        # node = inverted_mapping[model_id]
        node = self.inverted_mapping[model_id]
        
        # self.eval()
        # with torch.no_grad():
        #     x = self(torch.tensor(list(modelId_mapping.keys()), dtype=torch.long),#torch.tensor([node], dtype=torch.long), 
        #              data.edge_index, 
        #              data.edge_weight)
        #     probs, indices = torch.topk(x, top_k, dim=0)
        weight = self.embedding.weight[node]
        probs, indices = torch.topk(weight, k=top_k, dim=0)
                
        # norm_probs = self._normalize_probability(probs).squeeze().tolist()
        indices = indices.flatten().tolist()
        model_names = [self.modelId_mapping[index] for index in indices]
        model_downloads = [self.downloads_mapping[index] for index in indices]
        output_dataframe = \
            self._output_as_dataframe(model_names, model_downloads, probs)
        
        return output_dataframe
    
    def save_model(self, path: str = 'model/gnn.pt'):
        project_directory = os.path.dirname(os.path.abspath('.'))
        filepath = os.path.join(project_directory, path)
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(recommender_system.state_dict(), filepath)
    
    @staticmethod
    def _normalize_probability(probability):
        return probability / torch.sum(probability)
    
    def _output_as_dataframe(self, model_names, model_downloads, norm_probs):
        df = pd.DataFrame({
            "modelId": model_names, 
            "downloads": model_downloads, 
            "score": norm_probs.detach().numpy()
            })
        
        return df
    

# if __name__ == "__main__":
#     from gnn_datapipeline import Datapipeline
    
#     datapipeline = Datapipeline()
#     data, model_id_to_node_idx, downloads_to_node_idx = datapipeline.main()
    
#     # Define paramters
#     in_channels = data.num_features
#     hidden_channels = 32
#     out_channels = 1
    
#     # Instantiate GNN
#     recommender_system = RecommenderSystem(model_id_to_node_idx, 
#                                            downloads_to_node_idx,
#                                            in_channels, 
#                                            hidden_channels, 
#                                            out_channels)
    
#     # Define training parameters
#     optimizer = torch.optim.Adam(recommender_system.parameters(), lr=0.01)
#     criterion = torch.nn.CrossEntropyLoss()
#     num_epochs = 20

#     # Train the model
#     recommender_system.train_model(
#         data, 
#         model_id_to_node_idx, 
#         optimizer, 
#         criterion, 
#         num_epochs
#         )
    
#     # Evaluate the model
#     recommender_system.evaluate_model(data, model_id_to_node_idx, criterion)
    
#     # Inference
#     model_id = "bert-base-uncased"
#     k = 30
#     output_dataframe = recommender_system.predict(#data
#                                                   model_id,
#                                                   # model_id_to_node_idx, 
#                                                   top_k=k)

#     path = 'model/gnn.pt'
#     recommender_system.save_model(path)
#     project_directory = os.path.dirname(os.path.abspath('.'))
#     filepath = os.path.join(project_directory, path)
    
#     mapping_path = os.path.join(project_directory, 'model/model_mapping.json')
#     downloads_path = os.path.join(project_directory, 'model/downloads_mapping.json')
#     mapping_dir = os.path.dirname(mapping_path)
#     downloads_dir = os.path.dirname(downloads_path)
#     if not os.path.exists(mapping_dir):
#         os.makedirs(mapping_dir)
#     if not os.path.exists(downloads_dir):
#         os.makedirs(downloads_dir)
    
#     with open(mapping_path, 'w') as f:
#         json.dump(model_id_to_node_idx, f)

#     with open(downloads_path, 'w') as f:
#         json.dump(downloads_to_node_idx, f)
    
    
#     with open(mapping_path, 'r') as f:
#         model_mapping = json.load(f)
#         model_mapping = {int(key): value for key, value in model_mapping.items()}
#     with open(downloads_path, 'r') as f:
#         downloads_mapping = json.load(f)
#         downloads_mapping = {int(key): value for key, value in downloads_mapping.items()}

#     # For bryan
#     import os
#     import json
#     import torch
    
#     project_dir = os.path.dirname(os.path.abspath('.'))
#     model_path = os.path.join(project_dir, 'model/gnn.pt')
#     mapping_path = os.path.join(project_dir, 'model/model_mapping.json')
#     downloads_path = os.path.join(project_dir, 'model/downloads_mapping.json')
    
#     with open(mapping_path, 'r') as f:
#         model_mapping = json.load(f)
#         model_mapping = {int(key): value for key, value in model_mapping.items()}
#     with open(downloads_path, 'r') as f:
#         downloads_mapping = json.load(f)
#         downloads_mapping = {int(key): value for key, value in downloads_mapping.items()}
    
#     loaded_model = RecommenderSystem(model_mapping, downloads_mapping, in_channels=2688, hidden_channels=32, out_channels=1)
#     loaded_model.load_state_dict(torch.load(filepath))
#     output_df = loaded_model.predict(model_id, 30)
    
#     # For walter
#     output_df['score'] = output_df['score'] / np.linalg.norm(output_df['score'])


        
