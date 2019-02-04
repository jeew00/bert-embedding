import pandas as pd
import numpy as np
from tqdm import trange
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from bert_serving.client import BertClient
import stringdist as sd
import thai_tokenization as tkn
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--nparray_path', required=True, type=str, help="Numpy array file (.npy) of pre-embed sentences path")
parser.add_argument('--excel_path', required=True, type=str, help="Excel dataset file path")
parser.add_argument('--thai_model', action='store_true', help="For thai model")
parser.add_argument('test_size', default=0.25, help="Test set size")
args = parser.parse_args()

random_state = 0

root = os.getcwd()
bc = BertClient()
df = pd.read_excel(args.excel_path)
df['bert'] = np.load(args.nparray_path).tolist()
train_set, test_set = train_test_split(df, test_size=args.test_size, shuffle=True, random_state=random_state)

feat = np.vstack(train_set['bert'].values)
nfeat = normalize(feat)

tokenizer = tkn.ThaiTokenizer(
    os.path.join(root, 'models', 'bert_base_th', 'th.wiki.bpe.op25000.vocab'),
    os.path.join(root, 'models', 'bert_base_th', 'th.wiki.bpe.op25000.model'))

def query_nearest(nsentence, thai_model=False):
    if args.thai_model:
        nsentence = tokenizer.tokenize(nsentence)
    qfeat = bc.encode([nsentence], is_tokenized=args.thai_model, show_tokens=False)
    qfeat = normalize(qfeat)
    resind = np.argmax(np.dot(nfeat,qfeat.T))
    return train_set.iloc[resind]['Topic']

def query_editdist(nsentence):
    distList = [sd.levenshtein_norm(str(d),nsentence) for d in train_set['Keyword'].values]
    resind = np.argmin(distList)
    return train_set.iloc[resind]['Topic']

def evaluate_test():
    bertRes = []
    editRes = []
    # cut_test_set = test_set
    print('Evaluate on test set: {} examples'.format(int(df.shape[0] * args.test_size)))
    for i in trange(len(test_set)):
        ts = test_set.iloc[i]['Keyword']
        bertRes.append(query_nearest(ts, thai_model=args.thai_model))
        editRes.append(query_editdist(ts))
    test_set['bert_result'] = bertRes
    test_set['edit_result'] = editRes
    #return cut_test_set
    accBert = sum(test_set['bert_result'] == test_set['Topic'])/len(test_set)
    accEdit = sum(test_set['edit_result'] == test_set['Topic'])/len(test_set)

    print('Random state:\t{}'.format(random_state))
    print("Train set's size: {} ({}%)".format(int(train_set.shape[0]), (1 - args.test_size) * 100))
    print("Test set's size: {} ({}%)".format(int(test_set.shape[0]), args.test_size * 100))
    print('{} acc: {}'.format(args.nparray_path.split('/')[-1], accBert))
    print('Edit acc: {}'.format(accEdit))
    bc.close()
    # return accBert,accEdit

evaluate_test()
