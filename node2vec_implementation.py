## data processing and getting the node2vec embedding
## edges dump, edges pass to node2vec to get the embedding
## embedding
## getting other data features
## concatenating them
## applying a classifier to the model
import os
import logging
import networkx as nx
import numpy as np
import pandas as pd
import subprocess
import scipy as sp
import scipy.sparse
from haversine import haversine
import sys
import argparse
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from dataprep import DataLoader, dump_obj, load_obj

### masking funcions
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def log_dir(directory):
    log_dir = os.path.join(directory, 'results')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def geo_eval(data, y_pred):
    adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_eval, classLatMedian, classLonMedian, userLocation,w_adj_s, edges, total_users= data

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

def preprocess_data(data_home,args, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    encoding = kwargs.get('encoding', 'iso-8859-1')
    celebrity_threshold = kwargs.get('celebrity', 10)
    mindf = kwargs.get('mindf', 10)
    d2v = kwargs.get('d2v', False)
    adj_d2v = args.adj_d2v

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

    elif args.word2vec:
        dl.word2vec(args)
        X_train = dl.X_train_word2vec
        X_dev = dl.X_dev_word2vec
        X_test = dl.X_test_word2vec

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
    edges = list(G.edges)
    edges = np.array(edges)
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


def node2vec_prep(args, data):
    adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian,userLocation,w_adj_s, edges, total_users = data

    if type(w_adj_s) != int:
        weighted = True
    else:
        weighted = False


    output_name = 'node2vec_output' + '_p_' + str(args.p) + '_q_' + str(args.q) + '_weighted_' + str(weighted * 1) + '.emb'

    output_address = os.path.join(args.dir, output_name)


    if os.path.exists(output_address) and not model_args.builddata:
        logging.info('loading node2vec data from dumped file ' + output_address)
        df = pd.read_table(output_address, sep=' ', skiprows=[0], header=None)
        logging.info('loading data finished!')
    else:
        edge_pd = pd.DataFrame(edges)
        input_address = os.path.join(args.dir, 'node2vec_input.txt')
        node2vec_file_address = os.path.join(os.getcwd(), 'node2vec/src/main.py')
        p = args.p
        q = args.q
        bashCommand = " python %s --input %s --output %s --p %d --q %d  " % (
            node2vec_file_address, input_address, output_address, p, q)

        if weighted:
            logging.info('using weighted node2vec')
            for i, row in edge_pd.iterrows():
                x_ind = row[0]
                y_ind  = row[1]
                edge_pd.loc[i, 'weights'] = int(w_adj_s[x_ind, y_ind]*10)+1


            edge_pd.weights = edge_pd.weights.astype(int)
            bashCommand = " python %s --input %s --output %s --p %d --q %d --weighted" % (node2vec_file_address,
                                                                                input_address,output_address, p, q)

        edge_pd.to_csv(input_address, index=False, sep=' ', header=False)
        logging.info('node2vec training started')
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        out, err = process.communicate()
        logging.info('node2vec output :')
        logging.info(out)

        logging.info('node2vec training finished')

        df = pd.read_table(output_address, sep=' ', skiprows=[0], header=None)

    df.set_index(df.columns[0], inplace = True)
    df.sort_index(inplace=True)  # sorting the edges for usage later as a matrix
    df = df.reindex(pd.RangeIndex(total_users)).fillna(0)  # filling the missing values for nodes with 0

    node2vec = df.values

    return node2vec


def main(data, node2vec, args, **kwargs):

    np.random.seed(args.seed)
    notoutput = kwargs.get('save_results')

    dtypeint = 'int32'
    adj, X_train, Y_train, X_dev, Y_dev, X_test, \
    Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation, w_adj_s, edges, total_users = data


    logging.info('stacking training, dev and test features and creating indices...')
    X = np.vstack([X_train, X_dev, X_test])
    if len(Y_train.shape) == 1:
        Y = np.hstack((Y_train, Y_dev, Y_test))
    else:
        Y = np.vstack((Y_train, Y_dev, Y_test))

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
    train_mask = sample_mask(train_indices, l)
    val_mask = sample_mask(dev_indices, l)
    test_mask = sample_mask(test_indices, l)

    concatenated_layer = sp.sparse.hstack([X, node2vec])
    concatenated_layer = concatenated_layer.tocsr()

    dataset_name = args.dir.split('/')[-1]

    filename = 'node2vec_results' + '_d_' + dataset_name + '_f_' + str(args.lblfraction[0]) + '_p_' + \
        str(args.p) + '_q_' + str(args.q) + '_d2v_' + str(args.doc2vec * 1) + '_d2vw_' + str(args.d2vwindow) + '_d2vdm_' + str(args.d2vdm) +  ".txt"
    fileaddress = os.path.join(log_dir(os.getcwd()), filename)

    if not notoutput:

        ### defining classifiers
        names = ["Neural Net"]

        if args.cv:
            parameters = {'solver': ['lbfgs'],
                          'max_iter': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
                          'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes': np.arange(10, 15),
                          'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
            classifiers = [GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)]

        else:
            classifiers = [MLPClassifier(alpha=1, max_iter = 2000)]


        ## there was too many classifiers, we only change it to perceptron because of performance superiority

        # names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
        #          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
        #          "Naive Bayes"]
        #
        # classifiers = [
        #     KNeighborsClassifier(3),
        #     SVC(kernel="linear", C=0.025),
        #     SVC(gamma=2, C=1),
        #     DecisionTreeClassifier(max_depth=5),
        #     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        #     MLPClassifier(alpha=1),
        #     AdaBoostClassifier(),
        #     GaussianNB()]


        results = pd.DataFrame(columns=['dataset','model_name', 'mean', 'median',
                                        'acc','score','p','q','fraction', 'd2v','window_size', 'd2vdm'])
        for name, clf in zip(names, classifiers):
            logging.info('Results of ' + name + ':')

            clf.fit(concatenated_layer[train_mask].toarray(), Y[train_mask])
            score = clf.score(concatenated_layer[test_mask].toarray(), Y[test_mask])
            y_predict = clf.predict(concatenated_layer[test_mask].toarray())
            mean, median, acc, distances, latlon_true, latlon_pred = geo_eval(data, y_predict)
            window_size = args.d2vwindow
            d2vdm = args.d2vdm
            row = [dataset_name, name, mean, median, acc,score, args.p, args.q, args.lblfraction[0], args.doc2vec*1, window_size, d2vdm]
            results.loc[-1] = row
            results.index = results.index + 1

        results = results.sort_index()

        results.to_csv(fileaddress)
        logging.info('results dumped to ' + fileaddress)
    elif os.path.exists(fileaddress):
        print('You have asked for not building the model again!')
        print('The results can be found in', fileaddress)
    else:
        print('The results file does not exist and you have asked not to build it!')






def parse_args(argv):


    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('-d2v', '--doc2vec', action='store_true', help='if exists use doc2vec instead of tfidf')
    parser.add_argument('-w2v', '--word2vec', action='store_true', help='if exists use word2vec instead of tfidf')
    parser.add_argument('-w2vvec', '--w2vvec', metavar='int', help='word2vec vector size', type=int, default=300)
    parser.add_argument('-w2vwindow', '--w2vwindow', metavar='int', help='word2vec window size', type=int, default=2)

    parser.add_argument('-cbow', '--cbow_mean', action='store_true',
                        help=' use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used')
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


    parser.add_argument('-builddata', action='store_true',
                        help='if exists do not reload dumped data, build it from scratch')

    parser.add_argument('-seed', metavar='int', help='random seed', type=int, default=77)
    parser.add_argument('-not_save_results', action='store_true',
                        help='if exists overwrite previous results for the same experiment', default=False)
    parser.add_argument('-lblfraction', nargs='+', type=float,
                        help="fraction of labelled data used for training e.g. 0.01 0.1", default=[1.0])
    args = parser.parse_args(argv)
    return args





if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    model_args = args
    data = preprocess_data(args=args, data_home=args.dir, encoding=args.encoding, celebrity=args.celebrity, bucket=args.bucket, mindf=args.mindf, d2v = args.doc2vec)
    node2vec = node2vec_prep(args, data

    main(data, node2vec, args, save_results=args.not_save_results)





