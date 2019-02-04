import argparse
import pandas as pd
import numpy as np
import pickle
import thai_tokenization as tkn
from bert_serving.client import BertClient


parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, help="MODEL = thai or multi")
parser.add_argument('--ckpt', required=True, type=int, help="Checkpoint number (int)")
parser.add_argument('--excel_path', required=True, type=str, help="path to excel dataset")
args = parser.parse_args()
def main():
    print('************** READING EXCEL **************')
    df = pd.read_excel(args.excel_path)
    keywords = df['Keyword'].tolist()
    is_thai = False
    print('************** DONE **************')
    if args.model == 'thai':
        print('************** TOKENIZING **************')
        is_thai = True
        tokenizer = tkn.ThaiTokenizer(
            './models/bert_base_th/th.wiki.bpe.op25000.vocab',
            './models/bert_base_th/th.wiki.bpe.op25000.model')
        keywords = list(map(tokenizer.tokenize, keywords))
        print('************** DONE **************')
    with BertClient(check_length=False) as bc:
        print('************** ENCODING **************')
        bert_embedded = bc.encode(keywords, is_tokenized=is_thai)
        print('************** DONE **************')
    print('************** SAVING **************')
    np.save('./datasets/botnoi-api/{}-ckpt-{}.npy'.format(args.model, args.ckpt), bert_embedded)
    print('************** DONE **************')

if __name__ == '__main__':
    assert (args.model == 'thai') or (args.model == 'multi')
    main()
