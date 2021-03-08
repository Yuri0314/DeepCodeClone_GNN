import argparse

from model.model import DCC_GNN
from preprocess.graph_preprocess import generate_ast, generate_graph
from preprocess.data_preprocsee import generate_pair_data
from preprocess.model_input import generate_model_input


def parse_args():
    parser = argparse.ArgumentParser(description='DeepCodeClone with GNN')
    parser.add_argument("--dataset", default="googlejam4_src",
                        help="which dataset to use.")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--input-dim", type=int, default=128,
                        help="dimension of input")
    parser.add_argument("--hidden-gnn-dim", type=int, default=128,
                        help="dimension of hidden layer in gnn")
    parser.add_argument("--graph-dim", type=int, default=128,
                        help="dimension of graph vector")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads used in gnn")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="number of hidden layers in gnn")
    parser.add_argument("--feat-drop", type=float, default=.2,
                        help="feature dropout in gnn")
    parser.add_argument("--attn-drop", type=float, default=.2,
                        help="attention dropout in gnn")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection in gnn")

    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    args = parser.parse_args()
    print(args)

    return args


def preprocess(args):
    generate_pair_data(args.dataset)
    file2ast, token2idx = generate_ast(args.dataset)
    file2graph, file2tokenIdx = generate_graph(file2ast, token2idx)
    train_data, test_data = generate_model_input(file2graph, file2tokenIdx, args.dataset)
    return train_data, test_data, len(token2idx)

def train(args, model, data):
    pass

if __name__ == '__main__':
    args = parse_args()
    train_data, test_data, token_size = preprocess(args)
    hidden_dims = [args.hidden_gnn_dim for _ in range(args.num_layers)]
    num_heads = [args.num_heads for _ in range(args.num_layers)]
    print(hidden_dims)
    print(num_heads)
    model = DCC_GNN(token_size, args.input_dim, hidden_dims, args.graph_dim, num_heads,
                    [128, 64], [32], 2, args.num_layers, args.feat_drop, args.attn_drop,
                    args.negative_slope, args.residual, 2, 2)
    print(model)
    print(token_size)
    # train(args, train_data)