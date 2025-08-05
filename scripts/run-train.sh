#!/usr/bin/env bash
set -euxo pipefail

# fine-tuning without stemming
python train.py --model-name sentence-transformers/all-mpnet-base-v2 --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-all-mpnet-base-v2
python train.py --model-name sentence-transformers/multi-qa-distilbert-cos-v1 --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-multi-qa-distilbert-cos-v1
python train.py --model-name sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-multi-qa-MiniLM-L6-cos-v1
python train.py --model-name sentence-transformers/msmarco-distilbert-cos-v5 --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-msmarco-distilbert-cos-v5
python train.py --model-name sentence-transformers/msmarco-MiniLM-L6-cos-v5 --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-msmarco-MiniLM-L6-cos-v5
python train.py --model-name StevenLimcorn/MelayuBERT --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-melayubert
python train.py --model-name indolem/indobert-base-uncased --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-indobert
python train.py --model-name ams-bow --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-bow.joblib
python train.py --model-name ams-tfidf --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-tfidf.joblib
python train.py --model-name ams-doc2vec --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-doc2vec.gensim --epochs 100
python train.py --model-name ams-fasttext --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-fasttext.gensim --epochs 100

# fine-tuning with stemming
python train.py --model-name sentence-transformers/all-mpnet-base-v2 --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-all-mpnet-base-v2-stem --stemming
python train.py --model-name sentence-transformers/multi-qa-distilbert-cos-v1 --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-multi-qa-distilbert-cos-v1-stem --stemming
python train.py --model-name sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-multi-qa-MiniLM-L6-cos-v1-stem --stemming
python train.py --model-name sentence-transformers/msmarco-distilbert-cos-v5 --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-msmarco-distilbert-cos-v5-stem --stemming
python train.py --model-name sentence-transformers/msmarco-MiniLM-L6-cos-v5 --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-msmarco-MiniLM-L6-cos-v5-stem --stemming
python train.py --model-name StevenLimcorn/MelayuBERT --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-melayubert-stem --stemming
python train.py --model-name indolem/indobert-base-uncased --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-indobert-stem --stemming
python train.py --model-name ams-bow-stem --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-bow-stem.joblib --stemming
python train.py --model-name ams-tfidf-stem --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-tfidf-stem.joblib --stemming
python train.py --model-name ams-doc2vec-stem --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-doc2vec-stem.gensim --stemming --epochs 100
python train.py --model-name ams-fasttext-stem --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-fasttext-stem.gensim --stemming --epochs 100

# fine-tuning with stemming extended
python train.py --model-name indolem/indobert-base-uncased --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-indobert-stem-50 --stemming --epochs 50
python train.py --model-name indolem/indobert-base-uncased --dataset-path ../data/dataset-new/cleaned/triplet.jsonl --output-path ../models/ams-indobert-stem-100 --stemming --epochs 100
