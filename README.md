# Bert Embedding and evaluation

## Quickstart
1. Create ```models``` directory and download all the models in it
2. Create `datasets/botnoi-api/` directory and download `botnoi_api_cleaned_v2.xlsx` in it
3. Download `th.wiki.bpe.op25000.vocab` and `th.wiki.bpe.op25000.model` and `vocab.txt` to `models/bert-base-th/`
4. 
```
pip install -r requirements.txt
```


## embed.py
Start bert serving server somewhere
```
python embed.py --model [thai, multi] --ckpt [int]
```
embed numpy array will be save at `datasets/botnoi-api/`

## evaluate.py
Start bert serving server with the same model that embed the numpy array
### Thai model
```
python evaluate.py --thai_model --nparray_path path/to/numpy/array --excel_path path/to/excel
```
### Multilingual model
```
python evaluate.py --nparray_path path/to/numpy/array --excel_path path/to/excel
```