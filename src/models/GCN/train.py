from tqdm import trange
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy import sparse
from utils import create_adjacency_matrix, normalize_adjacency_matrix
from models.GCN.gcn import GCN

class Trainer:
    def __init__(self, args, graph, features, target):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        A = create_adjacency_matrix(graph)
        self.A = normalize_adjacency_matrix(A, sparse.eye(A.shape[0]))
        self.create_model()
        self.train_test_split()
        self.transfer_node_sets()
        self.process_features()
        self.transfer_features()

    def create_model(self):
        """
        Defining a model and transfering it to GPU/CPU.
        """
        self.node_count = self.graph.number_of_nodes()
        self.number_of_labels = np.max(self.target)+1
        self.number_of_features = max([f for _, feats  in self.features.items() for f in feats])+1

        self.model = GCN(self.args,
                                self.number_of_features,
                                self.number_of_labels)

        self.model = self.model.to(self.device)

    def train_test_split(self):
        """
        Creating a train/test split.
        """
        random.seed(self.args.seed)
        nodes = [node for node in range(self.node_count)]
        random.shuffle(nodes)
        self.train_nodes = nodes[0:self.args.train_size]
        self.test_nodes = nodes[self.args.train_size:self.args.train_size+self.args.test_size]
        self.validation_nodes = nodes[self.args.train_size+self.args.test_size:]

    def transfer_node_sets(self):
        """
        Transfering the node sets to the device.
        """
        self.train_nodes = torch.LongTensor(self.train_nodes).to(self.device)
        self.test_nodes = torch.LongTensor(self.test_nodes).to(self.device)
        self.validation_nodes = torch.LongTensor(self.validation_nodes).to(self.device)

    def process_features(self):
        num_rows = max(map(int, self.features.keys()))+1
        num_cols = max(map(max, self.features.values()))+1
        dense_mx = torch.zeros((num_rows, num_cols))
        for row_str, cols in self.features.items():
            row = int(row_str)
            dense_mx[row, cols] = 1
        self.dense_mx = torch.FloatTensor(dense_mx)
         # CSR 행렬을 COO 행렬로 변환
        adj_coo = self.A.tocoo()

        # COO 행렬로부터 PyTorch 희소 텐서 생성
        indices = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
        values = torch.tensor(adj_coo.data, dtype=torch.float)
        shape = torch.Size(adj_coo.shape)
        self.A = torch.sparse_coo_tensor(indices, values, shape)
        self.target = torch.LongTensor(self.target)
    def transfer_features(self):
        """
        Transfering the features and the target matrix to the device.
        """
        self.target = self.target.to(self.device)
        self.dense_mx = self.dense_mx.to(self.device)
        self.A = self.A.to(self.device)
    def score(self, index_set):
        """
        Calculating the accuracy for a given node set.
        :param index_set: Index of nodes to be included in calculation.
        :parm acc: Accuracy score.
        """
        self.model.eval()
        _, pred = self.model(self.dense_mx, self.A).max(dim=1)
        correct = pred[index_set].eq(self.target[index_set]).sum().item()
        acc = correct / index_set.size()[0]
        return acc

    def do_a_step(self):
        """
        Doing an optimization step.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        prediction = self.model(self.dense_mx, self.A)
        loss = F.nll_loss(prediction[self.train_nodes],
                                            self.target[self.train_nodes])
        loss.backward()
        self.optimizer.step()
    def train_neural_network(self):
        """
        Training a neural network.
        """
        print("\nTraining.\n")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.best_accuracy = 0
        self.step_counter = 0
        iterator = trange(self.args.epochs, desc='Validation accuracy: ', leave=True)
        for _ in iterator:
            self.do_a_step()
            accuracy = self.score(self.validation_nodes)
            iterator.set_description("Validation accuracy: {:.4f}".format(accuracy))
            if accuracy >= self.best_accuracy:
                self.best_accuracy = accuracy
                self.test_accuracy = self.score(self.test_nodes)
                self.step_counter = 0
            else:
                self.step_counter = self.step_counter + 1
                if self.step_counter > self.args.early_stopping_rounds:                
                    iterator.close()
                    print("\nBreaking from training process because of early stopping.\n")
                    break

    def fit(self):
        """
        Fitting the network and calculating the test accuracy.
        """
        self.train_neural_network()
        print("Test accuracy: {:.4f}".format(self.test_accuracy))