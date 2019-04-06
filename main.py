
## READING THE DATA
import os

import torch

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.nn import GATConv

import torch

from torch_geometric.data import Data

import logging


from torch_geometric.nn import ARMAConv

import torch.nn.functional as F
from collections import Counter
import numpy as np

import matplotlib
from sklearn.preprocessing.data import StandardScaler
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn
seaborn.set_style('white')
import sys
import numpy as np
import argparse
import pickle
import gzip
import os
from datetime import datetime
from haversine import haversine

import networkx as nx
import scipy as sp
import logging
from dataprep import DataLoader, dump_obj, load_obj

def log_dir(directory):
    log_dir = os.path.join(directory, 'results')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

### masking funcions
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def mask(mask_idx,l):
    mask = sample_mask(mask_idx,l)
    mask = torch.tensor(mask*1, dtype=torch.uint8)
    return mask


def preprocess_data(data_home, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    encoding = kwargs.get('encoding', 'iso-8859-1')
    celebrity_threshold = kwargs.get('celebrity', 10)
    mindf = kwargs.get('mindf', 10)
    d2v =  kwargs.get('d2v', False)


    one_hot_label = kwargs.get('onehot', False)
    vocab_file = os.path.join(data_home, 'vocab.pkl')
    if d2v:
        dump_name  = 'doc2vec_dump.pkl'
    else:
        dump_name = 'dump.pkl'
    dump_file = os.path.join(data_home,dump_name)
    if os.path.exists(dump_file) and not model_args.builddata:
        logging.info('loading data from dumped file ' + dump_name)
        data = load_obj(dump_file)
        logging.info('loading data finished!')
        return data

    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding,
                    celebrity_threshold=celebrity_threshold, one_hot_labels=one_hot_label, mindf=mindf,
                    token_pattern=r'(?u)(?<![@])#?\b\w\w+\b')

    dl.load_data()
    dl.assignClasses()
    if d2v:
        dl.doc2vec()
        X_train = dl.X_train_doc2vec
        X_test = dl.X_test_doc2vec
        X_dev = dl.X_dev_doc2vec
    else:
        dl.tfidf()
        X_train = dl.X_train
        X_dev = dl.X_dev
        X_test = dl.X_test
        vocab = dl.vectorizer.vocabulary_
        logging.info('saving vocab in {}'.format(vocab_file))
        dump_obj(vocab, vocab_file)
        logging.info('vocab dumped successfully!')


    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()




    dl.get_graph()
    logging.info('creating adjacency matrix...')
    adj = nx.adjacency_matrix(dl.graph, nodelist=range(len(U_train + U_dev + U_test)), weight='w')
    G = dl.graph
    # converting the edges index to pytorch format


    edges = list(G.edges)
    edges_test_hash = set(edges)  # used to make search faster

    for index, item in enumerate(edges):
        swapped = (item[1], item[0])
        if swapped not in edges_test_hash:
            edges.append(swapped)

    edges = sorted(edges)
    edges = np.array(edges)

    logging.info('adjacency matrix created.')


    Y_test = dl.test_classes
    Y_train = dl.train_classes
    Y_dev = dl.dev_classes
    classLatMedian = {str(c): dl.cluster_median[c][0] for c in dl.cluster_median}
    classLonMedian = {str(c): dl.cluster_median[c][1] for c in dl.cluster_median}

    P_test = [str(a[0]) + ',' + str(a[1]) for a in dl.df_test[['lat', 'lon']].values.tolist()]
    P_train = [str(a[0]) + ',' + str(a[1]) for a in dl.df_train[['lat', 'lon']].values.tolist()]
    P_dev = [str(a[0]) + ',' + str(a[1]) for a in dl.df_dev[['lat', 'lon']].values.tolist()]
    userLocation = {}
    for i, u in enumerate(U_train):
        userLocation[u] = P_train[i]
    for i, u in enumerate(U_test):
        userLocation[u] = P_test[i]
    for i, u in enumerate(U_dev):
        userLocation[u] = P_dev[i]

    data = (adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian,
            userLocation, edges)
    if not model_args.builddata:
        logging.info('dumping data in {} ...'.format(str(dump_file)))
        dump_obj(data, dump_file)
        logging.info('data dump finished!')

    return data



## my code

def data_loader(edges,features,y):

    """Data_loader takes the inputs in NumPy and SciPy format. It outputs data and number of classes
    inputs:
        edges: Graph connectivity in COO format with shape
        features: Node feature matrix
        y: labels in numpy array format

    """


    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    x = torch.tensor(features.todense(), dtype=torch.float)

    y = torch.tensor(y)

    data = Data(x=x, edge_index=edge_index, y = y)

    return data


def model_selection(model_name,data_torch, dropout, hidden_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'GraphSAGE':
        model, data = GraphSAGE_Net(dropout=dropout, layer_size= hidden_size, data_torch= data_torch).to(device), data_torch.to(device)
    if model_name == 'ARMA':
        model, data = ARMA_Net(dropout=dropout, layer_size= hidden_size, data_torch= data_torch).to(device), data_torch.to(device)
    if model_name == 'GAT':
        model, data = GAT_Net(dropout=dropout, layer_size=hidden_size, data_torch=data_torch).to(
            device), data_torch.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    return model, data, optimizer





def geo_eval(data, y_pred):
    adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_eval, classLatMedian, classLonMedian, userLocation, edges = data

    assert len(y_pred) == len(U_eval), "#preds: %d, #users: %d" % (len(y_pred), len(U_eval))
    distances = []
    latlon_pred = []
    latlon_true = []
    for i in range(0, len(y_pred)):
        user = U_eval[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        latlon_true.append([lat, lon])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
        latlon_pred.append([lat_pred, lon_pred])
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))

    logging.info(
        "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(
            int(acc_at_161)))

    return np.mean(distances), np.median(distances), acc_at_161, distances, latlon_true, latlon_pred


class GraphSAGE_Net(torch.nn.Module):

    def __init__(self, layer_size, dropout, data_torch):
        super(GraphSAGE_Net, self).__init__()
        self.layer_size = layer_size
        self.dropout = dropout
        self.num_hidden_layers = len(self.layer_size)
        self.data = data_torch
        model_name = '= SAGEConv('
        self.num_classes = int(self.data.y.max() + 1)

        # I used this bizarre coding to help it be more flexible for multiple structures :)
        for i in range(self.num_hidden_layers):
            layer_name = 'conv' + str(i)
            if i == 0:
                command = 'self.' + layer_name + model_name+ 'self.data.num_features,' + str(layer_size[i]) + ')'
                exec(command)
            elif i == (self.num_hidden_layers-1):
                command = 'self.' + layer_name + model_name + str(layer_size[i-1]) + ',' + str(self.num_classes) + ')'
                exec(command)
            else:
                command = 'self.' + layer_name + model_name + str(layer_size[i-1]) + ',' + str(layer_size[i]) + ')'
                exec(command)


    def forward(self):
        self.x2, self.edge_index2 = self.data.x, self.data.edge_index
        for i in range(self.num_hidden_layers):
            command = 'self.x2 = F.relu(self.conv' + str(i) + '(self.x2, self.edge_index2))'
            exec(command)

        self.x2 = F.dropout(self.x2, p=self.dropout, training=self.training)
        return F.log_softmax(self.x2, dim=1)







class ARMA_Net(torch.nn.Module):
    def __init__(self, dropout, layer_size,data_torch):
        '''It has only two layers because multiple layers will cause addition of noise.'''

        super(ARMA_Net, self).__init__()
        self.num_layers = 3
        self.dropout = dropout
        self.layer_size = layer_size

        self.data = data_torch
        self.num_classes = int(self.data.y.max() + 1)

        self.conv1 = ARMAConv(
            self.data.num_features,
            self.layer_size[0],
            num_stacks=3,
            num_layers=self.num_layers,
            shared_weights=True,
            dropout=self.dropout)

        self.conv2 = ARMAConv(
            self.layer_size[1],
            self.num_classes,
            num_stacks=3,
            num_layers=self.num_layers,
            shared_weights=True,
            dropout=self.dropout)

    def forward(self):
        x, edge_index = self.data.x, self.data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)





class GAT_Net(torch.nn.Module):
    def __init__(self, dropout, layer_size,data_torch):
        self.num_layers = 3
        self.dropout = dropout
        self.layer_size = layer_size
        self.data = data_torch
        self.num_classes = int(self.data.y.max() + 1)
        super(GAT_Net, self).__init__()
        self.att1 = GATConv(self.data.num_features, self.layer_size[0], heads=self.layer_size[0], dropout=self.dropout)
        self.att2 = GATConv(self.layer_size[1], self.num_classes, dropout=self.dropout)

    def forward(self):
        x = F.dropout(self.data.x, p=self.dropout, training=self.training)
        x = F.elu(self.att1(x, self.data.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.att2(x, self.data.edge_index)
        return F.log_softmax(x, dim=1)





def train(train_mask,model,optimizer,y):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[train_mask], y[train_mask]).backward()
    optimizer.step()


def test(masks,model,y):
    model.eval()
    logits, accs = model(), []
    for mask in masks:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def predict(mask, model,  args, **kwargs):
    output = kwargs.get('save_results')
    model.eval()
    logits = model()
    y_pred = logits[mask].max(1)[1]
    mean, median, acc, distances, latlon_true, latlon_pred = geo_eval(data, np.array(y_pred.cpu()))
    list_layers = kwargs.get('hidden')
    string = str()
    for i in list_layers:
        string = string  + str(i)
        string = string + '_'


    if output:
        with open(log_dir(os.getcwd()) +
                  "/results_" + kwargs.get('model_name') + '_l_' + string + '_d_'
                  + str(kwargs.get('dropout')) + '_f_' + str(kwargs.get('lblfractions')) + ".txt", "w") as fp:
            fp.write("mean={:.1f} , median={:.1f} acc={:.1f}".
                     format(mean, median, acc))








def main(data, args, **kwargs):
    hidden_size = kwargs.get('hidden', [400,400])
    dropout = kwargs.get('dropout', 0.0)
    model_name = kwargs.get('model_name', 'graphsage')

    dtypeint = 'int32'
    check_percentiles = kwargs.get('percent', False)
    adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian,userLocation, edges = data

    ## passing the data to pytorch format


    logging.info('stacking training, dev and test features and creating indices...')
    X = sp.sparse.vstack([X_train, X_dev, X_test])
    if len(Y_train.shape) == 1:
        Y = np.hstack((Y_train, Y_dev, Y_test))
    else:
        Y = np.vstack((Y_train, Y_dev, Y_test))

    verbose = not args.silent
    fractions = args.lblfraction
    stratified = False
    all_train_indices = np.asarray(range(0, X_train.shape[0])).astype(dtypeint)

    for percentile in fractions:
        logging.info('***********percentile %f ******************' % percentile)
        if stratified:
            all_chosen = []
            for lbl in range(0, np.max(Y_train) + 1):
                lbl_indices = all_train_indices[Y_train == lbl]
                selection_size = int(percentile * len(lbl_indices)) + 1
                lbl_chosen = np.random.choice(lbl_indices, size=selection_size, replace=False).astype(dtypeint)
                all_chosen.append(lbl_chosen)
            train_indices = np.hstack(all_chosen)
        else:
            selection_size = min(int(percentile * X.shape[0]), all_train_indices.shape[0])
            train_indices = np.random.choice(all_train_indices, size=selection_size, replace=False).astype(dtypeint)
        num_training_samples = train_indices.shape[0]
        logging.info('{} training samples'.format(num_training_samples))
        # train_indices = np.asarray(range(0, int(percentile * X_train.shape[0]))).astype(dtypeint)
        dev_indices = np.asarray(range(X_train.shape[0], X_train.shape[0] + X_dev.shape[0])).astype(dtypeint)
        test_indices = np.asarray(
            range(X_train.shape[0] + X_dev.shape[0], X_train.shape[0] + X_dev.shape[0] + X_test.shape[0])).astype(
            dtypeint)
    l = X.shape[0]
    train_mask = mask(train_indices, l)
    val_mask = mask(dev_indices, l)
    test_mask = mask(test_indices, l)
    masks = [train_mask,val_mask, test_mask]


    data_2= data_loader(edges,X,Y)
    print(model_name)
    model, data_2, optimizer = model_selection(model_name, data_torch=data_2, dropout=dropout, hidden_size=hidden_size)

    best_val_acc = test_acc = 0

    for epoch in range(1, 201):
        train(train_mask,model,optimizer,data_2.y)
        train_acc, val_acc, tmp_test_acc = test(masks,model,data_2.y)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'

        logging.info(log.format(epoch, train_acc, best_val_acc, test_acc))
    # predicting in terms of geographic measures
    predict(test_mask,model, args, save_results =args.save_results,hidden=args.hid, dropout=args.dropout, model_name= args.model_name, fractions = args.lblfraction)




def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', metavar='str', help='select model type from GraphSAGE, ARMA or GAT '
                                                                   'default GraphSAGE', type=str, default='GraphSAGE')
    parser.add_argument('-d2v', '--doc2vec', action='store_true', help='if exists use doc2vec instead of tfidf')


    parser.add_argument('-bucket', '--bucket', metavar='int', help='discretisation bucket size', type=int, default=300)
    parser.add_argument('-hid', nargs='+', type=int, help="list of hidden layer sizes", default=[100])
    parser.add_argument('-mindf', '--mindf', metavar='int', help='minimum document frequency in BoW', type=int,
                        default=10)
    parser.add_argument('-d', '--dir', metavar='str', help='home directory', type=str, default='./datasets/cmu')
    parser.add_argument('-enc', '--encoding', metavar='str', help='Data Encoding (e.g. latin1, utf-8)', type=str,
                        default='iso-8859-1')
    parser.add_argument('-reg', '--regularization', metavar='float', help='regularization coefficient)', type=float,
                        default=1e-6)
    parser.add_argument('-cel', '--celebrity', metavar='int', help='celebrity threshold', type=int, default=10)

    parser.add_argument('-dropout', type=float, help="dropout value default(0)", default=0)
    parser.add_argument('-percent', action='store_true', help='if exists loop over different train/dev proportions')
    parser.add_argument('-builddata', action='store_true',
                        help='if exists do not reload dumped data, build it from scratch')
    parser.add_argument('-lp', action='store_true', help='if exists use label information')
    parser.add_argument('-notxt', action='store_false', help='if exists do not use text information')
    parser.add_argument('-maxdown', help='max iter for early stopping', type=int, default=10)
    parser.add_argument('-silent', action='store_true', help='if exists be silent during training')
    parser.add_argument('-seed', metavar='int', help='random seed', type=int, default=77)
    parser.add_argument('-save', action='store_true', help='if exists save the model after training')
    parser.add_argument('-save_results', action='store_true', help='if exists save the results on the test set with geo eval')
    parser.add_argument('-load', action='store_true', help='if exists load pretrained model from file')
    parser.add_argument('-lblfraction', nargs='+', type=float,
                        help="fraction of labelled data used for training e.g. 0.01 0.1", default=[1.0])
    args = parser.parse_args(argv)
    return args




if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    model_args = args

    data = preprocess_data(data_home=args.dir, encoding=args.encoding, celebrity=args.celebrity,
                                          bucket=args.bucket,mindf=args.mindf, d2v = args.doc2vec)

    main(data, args, hidden=args.hid, regularization=args.regularization, dropout=args.dropout, model_name= args.model_name,
         percent=args.percent)
