import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run Trainer.")
    parser.add_argument("--edge-path",
                        nargs="?",
                        default="/home/daewon/gcn-template/datasets/Cora/preprocessed/cora_edges.csv",
                        help="Edge list csv.")
    parser.add_argument("--feature-path",
                        nargs="?",
                        default="/home/daewon/gcn-template/datasets/Cora/preprocessed/cora_features.json",
                        help="Features json.")
    parser.add_argument("--target-path",
                        nargs="?",
                        default="/home/daewon/gcn-template/datasets/Cora/preprocessed/cora_target.csv",
                        help="Target csv")
    parser.add_argument("--model",
                        nargs="?",
                        default="gcn",
                        help="Model type.")
    parser.add_argument("--epochs",
                        type=int,
                        default=2000,
                        help="# of Trainig epochs. Default is 2000.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for train-test split. Default is 42")
    parser.add_argument("--iterations",
                        type=int,
                        default=10,
                        help="Number of Approximate Persinalized PageRank iterations. Default is 10.")
    parser.add_argument("--early-stopping-rounds",
                        type=int,
                        default=500,
                        help="# of Training rounds before early stopping. Default is 10.")
    parser.add_argument("--train-size",
                        type=int,
                        default=1600,
                        help="Trainig set size. Default is 1600.")
    parser.add_argument("--val-size",
                        type=int,
                        default=200,
                        help="Validation set size. Default is 200.")
    parser.add_argument("--test-size",
                        type=int,
                        default=200,
                        help="Test set size. Default is 200.")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="Dropout parameter. Default is 0.5.")
    parser.add_argument("--alpha",
                        type=float,default=0.1,
                        help="Page rank teleport parameter. Default is 0.1.")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=1e-2,
                        help="Learning rate. Default is 0.01.")
    parser.add_argument("--lambd",
                        type=float,
                        default=0.005,
                        help="Weight matrix regularization. Default is 0.005.")
    parser.add_argument("--layers",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space. E.g. 64 64.")
    parser.add_argument("--weight-decay",
                        type=float,
                        default=5e-4,
                        help="Weight decay parameter. Default is 5e-4")
    parser.set_defaults(layers=[64, 64])

    return parser.parse_args()
