import argparse
import os
import pandas as pd
import numpy as np
import pickle
import thai_tokenization as tkn
from bert_serving.client import BertClient


parser = argparse.ArgumentParser()
parser.add_argument('--model_lang', required=True, type=str, help="MODEL = thai or multi")
parser.add_argument('--ckpt', required=True, type=int, help="Checkpoint number (int)")
args = parser.parse_args()

def main():
    root = os.getcwd()
    print('************** READING EXCEL **************')
    df = pd.read_excel(os.path.join('datasets', 'botnoi-api', 'botnoi_api_cleaned_v2.xlsx'))
    keywords = df['Keyword'].tolist()
    is_thai = False
    print('************** DONE **************')
    if args.model_lang == 'thai':
        print('************** TOKENIZING **************')
        is_thai = True
        tokenizer = tkn.ThaiTokenizer(
            os.path.join(root, 'models', 'bert_base_th', 'th.wiki.bpe.op25000.vocab'),
            os.path.join(root, 'models', 'bert_base_th', 'th.wiki.bpe.op25000.model'))
        keywords = list(map(tokenizer.tokenize, keywords))
        print('************** DONE **************')
    with BertClient(check_length=False) as bc:
        print('************** ENCODING **************')
        bert_embedded = bc.encode(keywords, is_tokenized=is_thai)
        print('************** DONE **************')
    print('************** SAVING **************')
    np.save(os.path.join(root, 'datasets', 'botnoi-api', '{}-ckpt-{}.npy'.format(args.model_lang, args.ckpt)), bert_embedded)
    print('************** DONE **************')

if __name__ == '__main__':
    assert (args.model_lang == 'thai') or (args.model_lang == 'multi')
    main()
