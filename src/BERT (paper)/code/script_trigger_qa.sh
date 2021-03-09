#!/bin/sh

export ACE_DIR=./data/ace-event/processed-data/default-settings/json


echo "=========================================================================================="
echo "                                          query 5                                         "
echo "=========================================================================================="

python code/run_trigger_qa.py \
  --train_file $ACE_DIR/train_convert.json \
  --dev_file $ACE_DIR/dev_convert.json  \
  --test_file $ACE_DIR/test_convert.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --eval_per_epoch 3 \
  --num_train_epochs 7 \
  --output_dir trigger_qa_output \
  --model_dir trigger_qa_output/epoch0-step0 \
  --learning_rate 4e-5 \
  --nth_query 5 \
  --warmup_proportion 0.1 \
  --do_train \
  --do_eval \
  --model bert-base-uncased 
  # To get final results on test, you need to (1) add eval_test; (2) set --model_dir to your model path for test (e.g., epoch-x-step-x)