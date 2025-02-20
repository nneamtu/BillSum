# BillSum

Code for the paper: [BillSum: A Corpus for Automatic Summarization of US Legislation](https://arxiv.org/abs/1910.00523) (Kornilova and Eidelman, 2019)

This paper will be presented at [EMNLP 2019 Workshop on New Frontiers in Summarization](https://summarization2019.github.io/)

[Link to slides from workshop](https://docs.google.com/presentation/d/1GEMSvUdS7lYo_WevKhSY0NuWzy6tm5IciCj0jq-r7Vc/edit?usp=sharing)

**Accessing the Dataset**: https://datahub.io/akornilo/billsum

Information on how the dataset was collected is under [BillSum_Data_Documentation.md](BillSum_Data_Documentation.md)

**NOTE**: The Rouge scores in the paper used the *clean* versions of the summaries as the reference summaries - (see Experiments section below). Using the original summaries should not make a significant difference, but may change the exact scores).

**Data Structure**
The data is stored in a jsonlines format, with one bill per line.

- text: bill text

- summary: (human-written) bill summary 

- title: bill title (can be used for generating a summary)

- bill_id: An identified for the bill - in US data it is SESSION_BILL-ID, for CA BILL-ID 


# Set-up

1. Install python dependencies (If using conda, use env.lst. If using pip, use requirements.txt)
2. Set the env `BILLSUM_PREFIX` to the base directory for all the data. (Download from link above)
3. Set `PYTHONPATH=.` to run code from this directory.
4. Install packages from `environment.lst` (we used conda, but you should be able to use pip
---

# Experiments

For all the experiments described in the paper, the texts were first cleaned using the script `billsum/data_prep/clean_text.py`. Results will be saved into the `BILLSUM_PREFIX/clean_final` directory.

## Sumy baselines

1. Clone [sumy](git@github.com:akornilo/sumy.git) and checkout the branch `ak_fork` (This is a minor modification on the original sumy library that allows it to work with my sentence selection logic).
2. In that directory run `pip install -e .`
3. From this directory, run `bill_sum/sumy_baselines.py`

## Supervised Experiments

### Preparing the data

1. Run `billsum/data_prep/clean_text.py` to clean up the whitespace formatting in the dataset. Outputs new jsonlines files with 'clean_text' field + original fields to `BILLSUM_PREFIX/clean_data`

2. Run `billsum/data_prep/label_sentences.py` to create labeled dataset.

This script takes each document, splits it into sentences, processes them with Spacy to get useful syntactic features and calculates the Rouge Score relative to the summary.

Outputs for each dataset part will be a pickle file with a dict of (bill_id, sentence data) pairs. (Stored under `PREFIX/sent_data/`) directory

```
Bill_id --> [
	('The monthly limitation for each coverage month during the taxable year is an amount equal to the lesser of 50 percent of the amount paid for qualified health insurance for such month, or an amount equal to 112 of in the case of self-only coverage, $1,320, and in the case of family coverage, $3,480. ',
	  [('The ', 186, 'the', '', 'O', 'DET', 'det', 188),
	   ('monthly ', 187, 'monthly', 'DATE', 'B', 'ADJ', 'amod', 188),
	   ('limitation ', 188, 'limitation', '', 'O', 'NOUN', 'nsubj', 197),
	   ...]
	  {'rouge-1': {'f': 0.2545454500809918,
	    'p': 0.3783783783783784,
	    'r': 0.1917808219178082},
	   'rouge-2': {'f': 0.09459459021183367, 'p': 0.14583333333333334, 'r': 0.07},
	   'rouge-l': {'f': 0.16757568176139123,
	    'p': 0.2972972972972973,
	    'r': 0.1506849315068493}}),
	    ...]
```

## Running Bert Models

0. Clone https://github.com/google-research/bert. Replace the `run_classifier.py` file with `billsum/bert_helpers/run_classifier.py` (adds custom code to read data in and out of files)

1. Create train.tsv / test.tsv files with `billsum/bert_helpers/prep_bert.py`. These will be stored under `PREFIX/bert_data` (set `$BERT_DATA_DIR` to point here)

2. Download the [Bert-Large, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip) model. 

3. Set `$BERT_BASE_DIR` environment variable to point to directory where you downloaded the model

3. Pretrain the Bert Model (run from the cloned bert repo)

```
python create_pretraining_data.py \
  --input_file=$BERT_DATA_DIR/all_texts_us_train.txt \
  --output_file=$BERT_DATA_DIR/all_texts_us_train.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```
```
python run_pretraining.py \
  --input_file=$BERT_DATA_DIR/all_texts_us_train.tfrecord\
  --output_dir=$BERT_MODEL_DIR \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```

`$BERT_MODEL_DIR` is the directory where you want to store your pretrained model.

This will take a while to run. 

4. To train the classifier model run (from bert repo):

``` 
python run_classifier.py   
			--task_name=simple
			--do_train=true   
			--do_predict=true   
			--do_predict_ca=true   
			--data_dir=$BERT_DATA_DIR   
			--vocab_file=$BERT_BASE_DIR/vocab.txt   
			--bert_config_file=$BERT_BASE_DIR/bert_config.json   
			--init_checkpoint=$BERT_MODEL_DIR/model.ckpt-20000   
			--max_seq_length=128  
			--train_batch_size=32   
			--num_train_epochs=3.0   
			--output_dir=$BERT_CLASSIFIER_DIR
```

Change `BERT_CLASSIFIER_DIR` to the directory where you want to store the classifier - should be different from pretraining directory. This script will create a model in the `BERT_CLASSIFIER_DIR` and store the sentence predictions in `BERT_CLASSIFIER_DIR/` dir.

5. Evaluate results using `bill_sum/bert_helpers/evaluate_bert.py`. Change the prefix variable to point to `BERT_CLASSIFIER_DIR` from above.

Results will be stored under `PREFIX/score_data/`


## Running feature classifier + ensemble

Run `bill_sum/train_wrapper.py`. Results will be stored under `BILLSUM_PREFIX/score_data/`

To get computations for the ensemble method run `billsum/evaluate_ensemble` (fix prefixes as before)

## Final Result aggregation

Run `billsum/compute_statistics.py` - will output summaries of results to terminal (also computers the oracle scores).


# Evaluate your own solution

If you have generated custom summaries for legislation, you can run `compute_rouge_from_texts.py` to evaluate your performance.

Update line 20 in the script to access files with your summaries.

It assumes each of these is a jsonlines file of the form:

```
{bill_id: 123, 'my_sum': 'This is my bill summary'}
{bill_id: 456, 'my_sum': 'Another summary here'}
```

It will print out the descriptive statistics for your method. If any of the summaries are longer than 2000 characters, it will throw an error - you can adjust this limit in the code.

The script will print as output descriptive statistics.

