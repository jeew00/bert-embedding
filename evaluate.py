import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
import os
import stringdist as sd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--nparray_path', required=True, type=str, help="Numpy array file (.npy) of pre-embed sentences path")
# parser.add_argument('--thai_model', action='store_true', help="For thai model")
# parser.add_argument('test_size', default=0.25, help="Test set size")
args = parser.parse_args()

random_state = 0

root = os.getcwd()
# bc = BertClient()
dataset = pd.read_excel(os.path.join('datasets', 'botnoi-api', 'botnoi_api_cleaned_v2.xlsx'))
dataset['bert'] = np.load(args.nparray_path).tolist()
# train_set, test_set = train_test_split(df, test_size=args.test_size, shuffle=True, random_state=random_state)



# feat = np.vstack(train_set['bert'].values)
# nfeat = normalize(feat)

# tokenizer = tkn.ThaiTokenizer(
#     os.path.join(root, 'models', 'bert_base_th', 'th.wiki.bpe.op25000.vocab'),
#     os.path.join(root, 'models', 'bert_base_th', 'th.wiki.bpe.op25000.model'))

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

def compute_accuracy(y_pred, y_true):
    correct = 0
    for pred_topic, test_topic in zip(y_pred, y_true):
        if pred_topic == test_topic:
            correct += 1
    return correct / len(y_pred) * 100

def eval_kfold_editdist(k):
    #################################################################
    #################################################################
    #### 5 Folds dataset v2                                      ####
    ####      Fold 1 Accuracy: 60.54567022538553                 ####
    ####      Fold 2 Accuracy: 61.05113299323763                 ####
    ####      Fold 3 Accuracy: 61.72736979475621                 ####
    ####      Fold 4 Accuracy: 60.32744097757741                 ####
    ####      Fold 5 Accuracy: 60.94435876141892                 ####
    ####      Average: 60.91919455047514                         ####
    #################################################################
    #################################################################
    print('Evaluating Edit Distance!')
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    accs = []
    fold_num = 1
    for train_indices, test_indices in kf.split(dataset):

        train_df = dataset.iloc[train_indices, :]
        train_keywords = train_df['Keyword'].values.tolist()
        train_topics = train_df['Topic'].values.tolist()

        test_df = dataset.iloc[test_indices, :]
        test_keywords = test_df['Keyword'].values.tolist()
        test_topics = test_df['Topic'].values.tolist()

        pred_topics = []
        for i, test_keyword in enumerate(test_keywords):
            distances = [sd.levenshtein(str(train_keyword), str(test_keyword)) for train_keyword in train_keywords]
            min_distance_idx = np.argmin(distances)
            pred_topic = train_topics[min_distance_idx]
            pred_topics.append(pred_topic)

        acc = compute_accuracy(pred_topics, test_topics)
        print('Fold {} accuracy: {}'.format(fold_num, acc))
        accs.append(acc)
        fold_num += 1
    avg_kfold_acc = mean(accs)
    print('Average {} fold accuracy: {}'.format(fold_num, avg_kfold_acc))
    return avg_kfold_acc

def eval_bert(k):
    print('Evaluating bert!')
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    accs = []
    fold_num = 1
    for train_indices, test_indices in kf.split(dataset):
        print('Fold #{}'.format(fold_num))
        train_df = dataset.iloc[train_indices, :]
        train_keywords = train_df['Keyword'].values.tolist()
        train_topics = train_df['Topic'].values.tolist()
        train_l2_embeds = normalize(np.vstack(train_df['bert'].values))
        
        test_df = dataset.iloc[test_indices, :]
        test_keywords = test_df['Keyword'].values.tolist()
        test_topics = test_df['Topic'].values.tolist()
        test_l2_embeds = normalize(np.vstack(test_df['bert'].values))

        pred_topics = []
        for test_l2_embed in tqdm(test_l2_embeds):
            closest_idx = np.argmax(np.dot(train_l2_embeds, test_l2_embed.T))
            pred_topic = train_topics[closest_idx]
            pred_topics.append(pred_topic)

        acc = compute_accuracy(pred_topics, test_topics)
        print('Fold {} accuracy: {:.2%}'.format(fold_num, acc))
        accs.append(acc)
        fold_num += 1
    avg_kfold_acc = mean(accs)
    print('Average {} fold accuracy: {:.2%}'.format(fold_num - 1, avg_kfold_acc))
    return avg_kfold_acc

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

if __name__ == '__main__':
    eval_bert(k=5)