import logging
import networkx as nx
import numpy as np
import numpy
from dataprep import DataLoader, dump_obj, load_obj
import scipy.sparse as sp
import json
from networkx.readwrite import json_graph
import os
import sys
import argparse
import subprocess
import gc

def process_dir(directory):
    log_dir = os.path.join(directory, 'graphsage_processed_data')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def json_dump(file, name):
    with open(name, 'w') as data_file:  ## dumping the json
        json.dump(file, data_file)


def default(o):
    if isinstance(o, np.int64): return int(o)
    if isinstance(o, numpy.int64): return int(o)
    if isinstance(o, np.float32): return float(o)
    if isinstance(o, np.int32): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    # if isinstance(o, np.ndarray): return o.tolist()
    # if isinstance(o, sp.csr.csr_matrix): return list(o)
    # else:
    #     raise TypeError('type is ', type(o))

### masking funcions
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def preprocess_data(data_home,args, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    encoding = kwargs.get('encoding', 'iso-8859-1')
    celebrity_threshold = kwargs.get('celebrity', 10)
    mindf = kwargs.get('mindf', 10)
    d2v = kwargs.get('d2v', False)
    adj_d2v = args.adj_d2v
    dtype = np.float32

    one_hot_label = kwargs.get('onehot', False)
    vocab_file = os.path.join(data_home, 'vocab.pkl')
    if d2v:
        dump_name = 'doc2vec_win_' + str(args.d2vwindow) + '_dm_' + str(args.d2vdm) + 'adj_d2v_'+ str(adj_d2v*1) + '_dump.pkl'
    else:
        dump_name = 'tfidf_win_' +  str(args.d2vwindow) + '_dm_' + str(args.d2vdm) + 'adj_d2v_' + str(adj_d2v*1) + '_dump.pkl'
    dump_file = os.path.join(data_home, dump_name)
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
        dl.doc2vec(args=args)
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

    if adj_d2v and args.doc2vec:
        adj = dl.adj_doc2vec
        G = nx.from_numpy_matrix(adj, parallel_edges=False, create_using=None)
    else:
        dl.get_graph()
        logging.info('creating adjacency matrix...')
        adj = nx.adjacency_matrix(dl.graph, nodelist=range(len(U_train + U_dev + U_test)), weight='w')
    # converting the edges index to pytorch format
        G = dl.graph


    edges = list(G.edges(data=True))

    edges = np.array(edges)
    edges = edges[:, :-1]

    edges = edges[np.lexsort(np.fliplr(edges).T)]


    wadj = args.weighted_adjacency ## if we want to weight adjacency materix

    if wadj:
        logging.info('multiplying weights...')
        w_adj_s = dl.adj_weight_d2v * adj
    else:
        w_adj_s = 0

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

    total_users = X_train.shape[0] + X_dev.shape[0] + X_test.shape[0]

    data = (adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian,userLocation, w_adj_s, edges, total_users)
    if not model_args.builddata:
        logging.info('dumping data in {} ...'.format(str(dump_file)))
        dump_obj(data, dump_file)
        logging.info('data dump finished!')

    return data

def graphsage_data_prep(data,args):
    adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian,userLocation, w_adj_s, edges, total_users = data
    dtypeint = np.int32
    dtypefloat = np.float32

    X_train = X_train.astype(dtype=dtypefloat)
    X_dev = X_dev.astype(dtype=dtypefloat)
    X_test = X_test.astype(dtype=dtypefloat)
    # changing dtype from npfloat64 to  np.float16 for reducing the memory footprint
    ## sparse matrix are not usable for graphsage json files

    # X_train_np = np.zeros(X_train.shape, X_train.dtype)
    # X_traino = X_train.tocoo()
    # X_train_np[X_traino.row, X_traino.col] = X_traino.data
    #
    # X_test_np = np.zeros(X_test.shape, X_test.dtype)
    # X_testo = X_test.tocoo()
    # X_test_np[X_testo.row, X_testo.col] = X_testo.data
    #
    # X_dev_np = np.zeros(X_dev.shape, X_dev.dtype)
    # X_devo = X_dev.tocoo()
    # X_dev_np[X_devo.row, X_devo.col] = X_devo.data
    #
    # del X_devo
    # del X_testo
    # del X_traino
    #
    #
    # X_dev = X_dev_np
    # X_train = X_train_np
    # X_test = X_test_np
    #
    # del X_dev_np
    # del X_train_np
    # del X_test_np
    
    X = sp.vstack([X_train, X_dev, X_test])  ## defining X as a dense
    ### fractions
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

    dataset_name = args.dir.split('/')[-1]     ## filenames definition
    train_prefix = dataset_name + '_f_' + str(percentile)
    ### check if files exist:
    class_id_file_name = train_prefix + '-class_map.json'
    if not os.path.exists(os.path.join(process_dir(os.getcwd()),class_id_file_name)):
        logging.info('creating the graphsage files')
        ## saving dense X to disk
        feat_name = train_prefix + '-feats'
        sp.save_npz(feat_name, X)
        ## defining y
        y = np.hstack((Y_train, Y_dev, Y_test))
        y_hot = np.zeros((total_users, y.max() + 1))
        y_hot[np.arange(total_users), y] = 1

        l = X.shape[0]  ## number of nodes
        ### train_mask, val_mask, and test mask for default setting
        train_mask = sample_mask(train_indices, l)
        val_mask = sample_mask(dev_indices, l)
        test_mask = sample_mask(test_indices, l)

        all_indices = np.hstack((train_indices,dev_indices,test_indices))
        ## creating the graph
        # creating labels
        test_keys = {k: v for (k, v) in enumerate(test_mask)}
        # train_keys = {k: v for (k, v) in enumerate(train_mask)}
        val_keys = {k: v for (k, v) in enumerate(val_mask)}
        label_keys = {k: v for (k, v) in enumerate(y_hot)}
        feature_keys = {}
        for idx, item in enumerate(X):
            feature_keys[idx] = item

        graphsage = nx.from_scipy_sparse_matrix(adj)
        nodes_dic = {}
        # nx.set_node_attributes(graphsage, nodes_list,'nodes')
        for i in all_indices:
            temp_dic = {}
            temp_dic['test'] = bool(test_keys[i])
            temp_dic['id'] = i
            # temp_dic['feature'] = (feature_keys[i]).tolist()
            temp_dic['val'] = bool(val_keys[i])
            # temp_dic['train'] = bool(train_keys[i])
            temp_dic['label'] = (label_keys[i]).tolist()
            nodes_dic[i] = temp_dic

        nx.set_node_attributes(G = graphsage, values=nodes_dic, name = None)
        # print(graphsage['nodes'])
        graphsage_json = json_graph.node_link_data(graphsage)
        for i in range(len(graphsage_json['nodes'])):
            value = list(graphsage_json['nodes'][i].values())[0]
            if type(value)!=int:
                graphsage_json['nodes'][i]= value

        graph_filename = train_prefix + '-G.json'
        with open(graph_filename, 'w') as data_file:  ## dumping the graph json
            json.dump(graphsage_json, data_file, default=default)

        id_map_file_name = train_prefix + '-id_map.json'
        # ## Creating ID map
        idMap = {str(i): int(i) for i in range(total_users)}
        #
        with open(id_map_file_name, 'w') as data_file:  ## dumping the json
            json.dump(idMap, data_file, default=default)

        class_map = {str(k): v for (k, v) in enumerate(y_hot.tolist())}

        with open(class_id_file_name, 'w') as data_file:  ## dumping the json, class_id_file_name already defined to check the existence of files
            json.dump(class_map, data_file,default=default)

        gc.collect()
        del graphsage_json
        del class_map
        del idMap

        feat_name = feat_name + '.npz'
        filenames = [feat_name, id_map_file_name, graph_filename, class_id_file_name]

        path = process_dir(os.getcwd())
        for f_name in filenames:
            os.rename(f_name, os.path.join(path,f_name))
    else:
        logging.info('graphsage processed files already exist')
        logging.info('They can be found at {}'.format(process_dir(os.getcwd())))




def run_main(args):
    path = process_dir(os.getcwd())
    dataset_name = args.dir.split('/')[-1]
    percentile = args.lblfraction[0]
    train_prefix = dataset_name + '_f_' + str(percentile)


    train_prefix = os.path.join(path,train_prefix)
    bashCommand = 'python -m graphsage.supervised_train --train_prefix %s --model %s --dropout %s --lblfraction %s --dataset_name %s' % (train_prefix, args.model_name, args.dropout, str(args.lblfraction[0]) ,dataset_name )
    logging.info('graphsage training started')
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    out, err = process.communicate()
    logging.info('graphsage output :')
    logging.info(out)
    logging.info('graphsage training finished')

def parse_args(argv):

    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('-d2v', '--doc2vec', action='store_true', help='if exists use doc2vec instead of tfidf')

    parser.add_argument('-bucket', '--bucket', metavar='int', help='discretisation bucket size', type=int, default=300)
    parser.add_argument('-p', '--p', metavar='int', help='p in node2vec algorithm', type=int, default=1)
    parser.add_argument('-q', '--q', metavar='int', help='q in node2vec algorithm', type=float, default=10)

    parser.add_argument('-hid', nargs='+', type=int, help="list of hidden layer sizes", default=[100])
    parser.add_argument('-mindf', '--mindf', metavar='int', help='minimum document frequency in BoW', type=int,
                        default=10)
    parser.add_argument('-d', '--dir', metavar='str', help='home directory', type=str, default='./datasets/cmu')
    parser.add_argument('-enc', '--encoding', metavar='str', help='Data Encoding (e.g. latin1, utf-8)', type=str,
                        default='iso-8859-1')

    parser.add_argument('-cel', '--celebrity', metavar='int', help='celebrity threshold', type=int, default=10)
    parser.add_argument('-d2vw', '--d2vwindow', metavar='int', help='d2vwindow', type=int, default=2)
    parser.add_argument('-d2vdm', '--d2vdm', metavar='int', help='d2v dm choice, 0 or 1 are valid', choices=[0,1], type=int, default=1)
    parser.add_argument('-wadj', '--weighted_adjacency', action='store_true',
                        help='use adjacency matrix with a weights derived from doc2vec similarities', default=False)
    parser.add_argument('-adj_d2v', '--adj_d2v', action='store_true',
                        help='use adjacency matrix derived from doc2vec similarities', default=False)
    parser.add_argument('-cv', '--cv', action='store_true', help='if exists apply grid search to neural nets', default=False)
    parser.add_argument('-dropout', type=float, help="dropout value default(0)", default=0)


    parser.add_argument('-builddata', action='store_true',
                        help='if exists do not reload dumped data, build it from scratch')

    parser.add_argument('-seed', metavar='int', help='random seed', type=int, default=77)
    parser.add_argument('-not_save_results', action='store_true',
                        help='if exists overwrite previous results for the same experiment', default=False)
    parser.add_argument('-lblfraction', nargs='+', type=float,
                        help="fraction of labelled data used for training e.g. 0.01 0.1", default=[1.0])
    parser.add_argument('-mn', '--model_name', metavar='str', choices=['graphsage_meanpool', 'graphsage_seq' , 'graphsage_mean', 'gcn'],
                        help='select model type from graphsage_meanpool, graphsage_seq, graphsage_mean, gcn '
                                                                   'default GraphSAGE', type=str, default='graphsage_mean')
    args = parser.parse_args(argv)
    return args



if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    model_args = args
    data = preprocess_data(args=model_args, data_home=args.dir, encoding=args.encoding, celebrity=args.celebrity, bucket=args.bucket, mindf=args.mindf, d2v = args.doc2vec)
    graphsage_data_prep(data, model_args)
    run_main(model_args)



