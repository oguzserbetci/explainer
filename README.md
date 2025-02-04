# Pipeline
Create config file for Baseline or Transformers under `config/`, using data classes defined under `train_baseline.py` and `train_transformers.py`, respectively. Then run the scripts in following order. Example of training using imdb sentiment analysis dataset is included.


Environment setup:
- conda create -n explainer python==3.11.9
- conda activate explainer
- pip install -r requirements.txt

Example setup:
- To use IMDB sentiment analysis, run `python -m src.create_imdb`

Data and task setup:
- Tasks need to be defined as a dictionary of `{'task name': number_of_classes}`, task name should not include `=` or `-`.
- Dataset samples should have an identifier column named `_id`.

Baseline:
1. `python -m train_baseline config/imdb_baseline.json`
2. `python -m extract_baseline_attr config/imdb_baseline.json`
3. `python -m agg_attributions -e tfidf -s test -o task -o pred results/imdb/checkpoint-#`

Transformers:
1. `CUDA_VISIBLE_DEVICES=0 python -m train_transformers config/imdb.json`
    - Use a single GPU because multiple GPUs will cause segments of text to be assigned to different GPUs.
2. `extract_transformers_attrs.py`
    - Extracts attributions from the finetuned transformer model.
    - `CUDA_VISIBLE_DEVICES=0 python extract_transformers_attrs.py -c config/imdb.json --split test -m lrp -d cuda results/imdb/checkpoint-#/`
6. `eval_attributions.py`
    - Token removal evaluation for LGXA, IG, LRP, and attention, including random for sanity check.
    - `CUDA_VISIBLE_DEVICES=0 python eval_attributions.py -N 500 -c config/imdb.json results/imdb/checkpoint-#/`
3. `postprocess_attrs.py`
    - Generate n-grams per sentence from the attributions.
    - `python postprocess_attrs.py -c config/imdb.json --split test -l en results/imdb/checkpoint-#`
4. `agg_attributions.py`
    - Creates ranked attributions and top 50 phrases per task.
    - `python agg_attributions.py -s test -o task -o pred results/imdb/checkpoint-#`
5. `attr_global_eval.py`
    - `python eval_attr_global.py -s test -c config/imdb.json results/imdb/checkpoint-# lrp_attrs/test_attr_pos_\!sentence_all_task-pred_avg_df.parquet`

# TODOS
- [ ] Allow other models to be used through the config.