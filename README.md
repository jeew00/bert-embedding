# Bert embedding and evaluation

## Quickstart
1. Create ```models``` directory and download all the models in it
2. Create `datasets/botnoi-api/` directory and download `botnoi_api_cleaned_v2.xlsx` in it
3. 
```
pip install -r requirements.txt
```


## embed.py
Start bert serving server somewhere
```
bert-serving-start [-h] -model_dir MODEL_DIR
               [-tuned_model_dir TUNED_MODEL_DIR]
               [-ckpt_name CKPT_NAME] [-config_name CONFIG_NAME]
               [-graph_tmp_dir GRAPH_TMP_DIR]
               [-max_seq_len MAX_SEQ_LEN]
               [-pooling_layer POOLING_LAYER [POOLING_LAYER ...]]
               [-pooling_strategy {NONE,REDUCE_MAX,REDUCE_MEAN,REDUCE_MEAN_MAX,FIRST_TOKEN,LAST_TOKEN}]
               [-mask_cls_sep] [-show_tokens_to_client]
               [-port PORT] [-port_out PORT_OUT]
               [-http_port HTTP_PORT]
               [-http_max_connect HTTP_MAX_CONNECT] [-cors CORS]
               [-num_worker NUM_WORKER]
               [-max_batch_size MAX_BATCH_SIZE]
               [-priority_batch_size PRIORITY_BATCH_SIZE] [-cpu]
               [-xla] [-fp16]
               [-gpu_memory_fraction GPU_MEMORY_FRACTION]
               [-device_map DEVICE_MAP [DEVICE_MAP ...]]
               [-prefetch_size PREFETCH_SIZE] [-verbose]
               [-version]
```
Open another terminal window then run
```
python embed.py --model_lang [thai, multi] --ckpt [int]
```
embed numpy array will be save at `datasets/botnoi-api/`

## Evaluate
See `evaluate.ipynb`