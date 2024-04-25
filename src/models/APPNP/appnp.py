import torch.nn as nn
import torch.nn.functional as F
from models.APPNP.layers import *


class APPNP(nn.Module):
    def __init__(self, args, number_of_labels, number_of_features, graph, device):
        super(APPNP, self).__init__()
        self.args = args
        self.number_of_labels = number_of_labels
        self.number_of_features = number_of_features
        self.graph = graph
        self.device = device
        self.setup_layers()
        self.setup_propagator()

    def setup_layers(self):
        self.layer1 = SparseFullyConnected(self.number_of_features, self.args.layers[0])
        self.layer2 = DenseFullyConnected(self.args.layers[1], self.number_of_labels)
        
    def setup_propagator(self):
        self.propagator = create_propagator_matrix(self.graph, self.args.alpha, self.args.model)
        if self.args.model == "exact":
            self.propagator = self.propagator.to(self.device)
        else:
            self.edge_indices = self.propagator["indices"].to(self.device)
            self.edge_weights = self.propagator["values"].to(self.device)

    def forward(self, features_indices, feature_values):
        feature_values = F.dropout(feature_values,
                                   p=self.args.dropout,
                                   training=self.training)
        
        latent_features_1 = self.layer1(features_indices, feature_values)
        latent_features_1 = F.relu(latent_features_1)
        latent_features_1 = F.dropout(latent_features_1,
                                      p=self.args.dropout,
                                      training=self.training)
        latent_features_2 = self.layer2(latent_features_1)
        if self.args.model == "exact":
            self.predictions = F.dropout(self.propagator,
                                         p=self.args.dropout,
                                         training=self.training)
            self.predictions = torch.mm(self.predictions, latent_features_2)
        else:
            localized_predictions = latent_features_2
            edge_weights = F.dropout(self.edge_weights,
                                    p=self.args.dropout,
                                    training=self.training)
            for iteration in range(self.args.iterations):

                new_features = spmm(index=self.edge_indices,
                                    value=edge_weights,
                                    n=localized_predictions.shape[0],
                                    m=localized_predictions.shape[0],
                                    matrix=localized_predictions)
                localized_predictions = (1-self.args.alpha)*new_features
                localized_predictions = localized_predictions + self.args.alpha*latent_features_2
            self.predictions = localized_predictions
        self.predictions = F.log_softmax(self.predictions, dim=1)
        return self.predictions