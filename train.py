import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os

from torch.utils.data import DataLoader
from tqdm import trange

from model.model import DCC_GNN
from preprocess.data_preprocsee import generate_pair_data, split_true_false_data
from preprocess.graph_preprocess import generate_ast, generate_graph
from preprocess.model_input import generate_model_input


def parse_args():
    parser = argparse.ArgumentParser(description='DeepCodeClone with GNN')
    parser.add_argument("--dataset", default="bigclonebenchdata",
                        help="which dataset to use.")
    parser.add_argument('--dataset-ratio', type=float, default=0.2,
                        help="how much data to use")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--input-dim", type=int, default=128,
                        help="dimension of input")
    parser.add_argument("--hidden-gnn-dim", type=int, default=128,
                        help="dimension of hidden layer in gnn")
    parser.add_argument("--graph-dim", type=int, default=128,
                        help="dimension of graph vector")
    parser.add_argument("--num-heads", type=int, default=3,
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

    parser.add_argument("--batch-size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument("--validation-split", type=float, default=0.2,
                        help="validation data ratio")
    parser.add_argument("--data-balance-ratio", type=int, default=1,
                        help="false data and true data balance ratio. Set -1 to not use balance")
    args = parser.parse_args()
    print(args)

    return args


def preprocess(args):
    if args.dataset == 'googlejam4_src':
        generate_pair_data('googlejam4_src')
    elif args.dataset == 'bigclonebenchdata':
        split_true_false_data('bigclonebenchdata')
    file2ast, token2idx = generate_ast(args.dataset)
    file2graph, file2tokenIdx = generate_graph(file2ast, token2idx)
    train_data, test_data = generate_model_input(file2graph, file2tokenIdx,
                                                 args.dataset, args.validation_split,
                                                 args.data_balance_ratio,
                                                 args.dataset_ratio)
    return train_data, test_data, len(token2idx)


def test(model, device, test_data, loss_func, epoch):
    dataloader = DataLoader(test_data, batch_size=128, shuffle=False,
                            collate_fn=lambda x: x, num_workers=4)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    loss = 0.0
    for _, batch in enumerate(dataloader):
        for x1, x2, label in batch:
            idx_list1, edges1, edge_types1 = x1
            idx_list1 = torch.tensor(idx_list1, dtype=torch.long, device=device)
            u1, v1 = zip(*edges1)
            u1 = torch.tensor(u1, dtype=torch.long, device=device)
            v1 = torch.tensor(v1, dtype=torch.long, device=device)
            edge_types1 = torch.tensor(edge_types1, dtype=torch.long, device=device)
            idx_list2, edges2, edge_types2 = x2
            idx_list2 = torch.tensor(idx_list2, dtype=torch.long, device=device)
            u2, v2 = zip(*edges2)
            u2 = torch.tensor(u2, dtype=torch.long, device=device)
            v2 = torch.tensor(v2, dtype=torch.long, device=device)
            edge_types2 = torch.tensor(edge_types2, dtype=torch.long, device=device)
            label_tensor = torch.tensor([label], dtype=torch.long, device=device)
            in_data = ([idx_list1, u1, v1, edge_types1], [idx_list2, u2, v2, edge_types2])
            output = model(in_data)
            loss = loss + loss_func(output, label_tensor)
            predict = torch.argmax(output, dim=1).item()
            if predict == 1 and label == 1:
                tp += 1
            if predict == 0 and label == 0:
                tn += 1
            if predict == 1 and label == 0:
                fp += 1
            if predict == 0 and label == 1:
                fn += 1

    loss = loss.item() / len(test_data)
    print('Test Loss=%g' % round(loss, 5))
    f_name = "./test_{}.log".format(args.dataset)
    if os.path.exists(f_name):
        f = open(f_name, 'a')
    else:
        f = open(f_name, 'w')
    f.write('Epoch {} '.format(epoch) + 'Loss=%g\n' % round(loss, 5))
    print(tp, tn, fp, fn)
    p = 0.0
    r = 0.0
    f1 = 0.0
    if tp + fp == 0:
        print('precision is none')
        f.close()
        return
    p = tp / (tp + fp)
    if tp + fn == 0:
        print('recall is none')
        f.close()
        return
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print('precision')
    print(p)
    print('recall')
    print(r)
    print('F1')
    print(f1)
    f.write('Precision: {}\n'.format(str(p)))
    f.write('Recall: {}\n'.format(str(r)))
    f.write('F1: {}\n'.format(str(f1)))
    f.flush()
    f.close()


def train(args, model, device, train_data, test_data, exist=-1):
    dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                            collate_fn=lambda x: x, num_workers=4)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_time = 0.0
    if exist == -1:
        f = open('./train_{}.log'.format(args.dataset), 'w')
        epochs = trange(args.epochs, desc='Epoch', leave=True)
    else:
        optimizer.load_state_dict(torch.load('./optimizer.pth'))
        f = open('./train_continue_{}.log'.format(args.dataset), 'w')
        epochs = trange(exist, args.epochs, desc='Epoch', leave=True)
    for epoch in epochs:
        start_time = time.time()
        model.train()
        total_loss = 0.0
        num = 0.0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch_loss = 0
            for x1, x2, label in batch:
                idx_list1, edges1, edge_types1 = x1
                idx_list1 = torch.tensor(idx_list1, dtype=torch.long, device=device)
                u1, v1 = zip(*edges1)
                u1 = torch.tensor(u1, dtype=torch.long, device=device)
                v1 = torch.tensor(v1, dtype=torch.long, device=device)
                edge_types1 = torch.tensor(edge_types1, dtype=torch.long, device=device)
                idx_list2, edges2, edge_types2 = x2
                idx_list2 = torch.tensor(idx_list2, dtype=torch.long, device=device)
                u2, v2 = zip(*edges2)
                u2 = torch.tensor(u2, dtype=torch.long, device=device)
                v2 = torch.tensor(v2, dtype=torch.long, device=device)
                edge_types2 = torch.tensor(edge_types2, dtype=torch.long, device=device)
                label = torch.tensor([label], dtype=torch.long, device=device)
                in_data = ([idx_list1, u1, v1, edge_types1], [idx_list2, u2, v2, edge_types2])
                output = model(in_data)
                batch_loss = batch_loss + loss_func(output, label)
            batch_loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += batch_loss.item()
            num += len(batch)
            loss = total_loss / num

            # 每100个batch记录一次loss
            if (i + 1) % 100 == 0:
                f.write("Epoch_{} ".format(epoch + 1) + "batch_{} ".format(str(i + 1)) +
                    "Training Loss=%g\n" % round(loss, 5))
                f.flush()
            epochs.set_description("Epoch {} ".format(epoch + 1) + "batch {} ".format(str(i + 1))
                                   + "(Training Loss=%g)" % round(loss, 5))

        epoch_time = time.time() - start_time
        f.write("Epoch_{} ".format(epoch + 1) + "Training time: %g\n" % round(epoch_time, 5))
        f.flush()
        train_time += epoch_time
        with torch.no_grad():
            test(model, device, test_data, loss_func, epoch)
        torch.save(model, './model_{}_{}.pth'.format(args.dataset, epoch + 1))
        torch.save(optimizer.state_dict(), './optimizer.pth')
    f.write("Total training time: %g\n" % round(train_time, 5))
    f.close()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1
                          else "cpu")
    train_data, test_data, token_size = preprocess(args)
    model = None
    for i in range(args.epochs - 1, -1, -1):
        f_name = './model_{}_{}.pth'.format(args.dataset, i + 1)
        if os.path.exists(f_name):
            model = torch.load(f_name, map_location=device)
            exist = i + 1
            break

    if model == None:
        # Define model structure
        hidden_dims = [args.hidden_gnn_dim for _ in range(args.num_layers - 1)]
        num_heads = [args.num_heads for _ in range(args.num_layers - 1)]
        model = DCC_GNN(token_size, args.input_dim, hidden_dims, args.graph_dim, num_heads,
                        [128, 64], [32], 2, args.num_layers, args.feat_drop, args.attn_drop,
                        args.negative_slope, args.residual, 2, 2)
        model.to(device)
        exist = -1
    train(args, model, device, train_data, test_data, exist=exist)
