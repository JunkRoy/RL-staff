---
dataset_info:
  features:
  - name: prompt
    dtype: string
  - name: chosen
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: rejected
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  splits:
  - name: descriptiveness
    num_bytes: 4730435
    num_examples: 5425
  - name: sentiment
    num_bytes: 4753415
    num_examples: 5480
  download_size: 6210965
  dataset_size: 9483850
configs:
- config_name: default
  data_files:
  - split: descriptiveness
    path: data/descriptiveness-*
  - split: sentiment
    path: data/sentiment-*
---
# TRL's Sentiment and Descriptiveness Preference Dataset

The dataset comes from https://arxiv.org/abs/1909.08593, one of the earliest RLHF work from OpenAI.

We preprocess the dataset using our standard `prompt, chosen, rejected` format.

## Reproduce this dataset

1. Download the `descriptiveness_sentiment.py` from the https://huggingface.co/datasets/trl-internal-testing/descriptiveness-sentiment-trl-style/tree/0.1.0.
2. Run `python examples/datasets/descriptiveness_sentiment.py --push_to_hub --hf_entity trl-internal-testing`
