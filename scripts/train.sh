#!/usr/bin/env bash
set -euxo pipefail

# --- eval pretrained models
python eval.py --model-name sentence-transformers/all-mpnet-base-v2 --dataset-path ../data/beir 
python eval.py --model-name sentence-transformers/multi-qa-distilbert-cos-v1 --dataset-path ../data/beir 
python eval.py --model-name sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --dataset-path ../data/beir 
python eval.py --model-name sentence-transformers/msmarco-distilbert-cos-v5 --dataset-path ../data/beir 
python eval.py --model-name sentence-transformers/msmarco-MiniLM-L6-cos-v5 --dataset-path ../data/beir 

python eval.py --model-name sentence-transformers/all-mpnet-base-v2 --dataset-path ../data/beir --stemming
python eval.py --model-name sentence-transformers/multi-qa-distilbert-cos-v1 --dataset-path ../data/beir --stemming
python eval.py --model-name sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --dataset-path ../data/beir --stemming
python eval.py --model-name sentence-transformers/msmarco-distilbert-cos-v5 --dataset-path ../data/beir --stemming
python eval.py --model-name sentence-transformers/msmarco-MiniLM-L6-cos-v5 --dataset-path ../data/beir --stemming

# --- fine tuning models
python train.py --model-name sentence-transformers/all-mpnet-base-v2 --dataset-path ../data/triplet/triplet.jsonl --output-path ../models/ams-all-mpnet-base-v2
python train.py --model-name sentence-transformers/multi-qa-distilbert-cos-v1 --dataset-path ../data/triplet/triplet.jsonl --output-path ../models/ams-multi-qa-distilbert-cos-v1
python train.py --model-name sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --dataset-path ../data/triplet/triplet.jsonl --output-path ../models/ams-multi-qa-MiniLM-L6-cos-v1
python train.py --model-name sentence-transformers/msmarco-distilbert-cos-v5 --dataset-path ../data/triplet/triplet.jsonl --output-path ../models/ams-msmarco-distilbert-cos-v5
python train.py --model-name sentence-transformers/msmarco-MiniLM-L6-cos-v5 --dataset-path ../data/triplet/triplet.jsonl --output-path ../models/ams-msmarco-MiniLM-L6-cos-v5
python train.py --model-name ams-bow --dataset-path ../data/triplet/triplet.jsonl --output-path ../models/ams-bow.joblib
python train.py --model-name ams-tf-idf --dataset-path ../data/triplet/triplet.jsonl --output-path ../models/ams-tf-idf.joblib

python train.py --model-name sentence-transformers/all-mpnet-base-v2 --dataset-path ../data/triplet/triplet.jsonl --output-path ../models/ams-all-mpnet-base-v2-stem --stemming
python train.py --model-name sentence-transformers/multi-qa-distilbert-cos-v1 --dataset-path ../data/triplet/triplet.jsonl --output-path ../models/ams-multi-qa-distilbert-cos-v1-stem --stemming
python train.py --model-name sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --dataset-path ../data/triplet/triplet.jsonl --output-path ../models/ams-multi-qa-MiniLM-L6-cos-v1-stem --stemming
python train.py --model-name sentence-transformers/msmarco-distilbert-cos-v5 --dataset-path ../data/triplet/triplet.jsonl --output-path ../models/ams-msmarco-distilbert-cos-v5-stem --stemming
python train.py --model-name sentence-transformers/msmarco-MiniLM-L6-cos-v5 --dataset-path ../data/triplet/triplet.jsonl --output-path ../models/ams-msmarco-MiniLM-L6-cos-v5-stem --stemming
python train.py --model-name ams-bow-stem --dataset-path ../data/triplet/triplet.jsonl --output-path ../models/ams-bow-stem.joblib
python train.py --model-name ams-tf-idf-stem --dataset-path ../data/triplet/triplet.jsonl --output-path ../models/ams-tf-idf-stem.joblib

# --- eval tuned models
python eval.py --model-name ../models/ams-all-mpnet-base-v2 --dataset-path ../data/beir
python eval.py --model-name ../models/ams-multi-qa-distilbert-cos-v1 --dataset-path ../data/beir
python eval.py --model-name ../models/ams-multi-qa-MiniLM-L6-cos-v1 --dataset-path ../data/beir
python eval.py --model-name ../models/ams-msmarco-distilbert-cos-v5 --dataset-path ../data/beir
python eval.py --model-name ../models/ams-msmarco-MiniLM-L6-cos-v5 --dataset-path ../data/beir
python eval.py --model-name ../models/ams-bow.joblib --dataset-path ../data/beir 
python eval.py --model-name ../models/ams-tf-idf.joblib --dataset-path ../data/beir 

python eval.py --model-name ../models/ams-all-mpnet-base-v2 --dataset-path ../data/beir --stemming
python eval.py --model-name ../models/ams-multi-qa-distilbert-cos-v1 --dataset-path ../data/beir --stemming
python eval.py --model-name ../models/ams-multi-qa-MiniLM-L6-cos-v1 --dataset-path ../data/beir --stemming
python eval.py --model-name ../models/ams-msmarco-distilbert-cos-v5 --dataset-path ../data/beir --stemming
python eval.py --model-name ../models/ams-msmarco-MiniLM-L6-cos-v5 --dataset-path ../data/beir --stemming
python eval.py --model-name ../models/ams-bow.joblib --dataset-path ../data/beir --stemming
python eval.py --model-name ../models/ams-tf-idf.joblib --dataset-path ../data/beir --stemming

python eval.py --model-name ../models/ams-all-mpnet-base-v2-stem --dataset-path ../data/beir
python eval.py --model-name ../models/ams-multi-qa-distilbert-cos-v1-stem --dataset-path ../data/beir
python eval.py --model-name ../models/ams-multi-qa-MiniLM-L6-cos-v1-stem --dataset-path ../data/beir
python eval.py --model-name ../models/ams-msmarco-distilbert-cos-v5-stem --dataset-path ../data/beir
python eval.py --model-name ../models/ams-msmarco-MiniLM-L6-cos-v5-stem --dataset-path ../data/beir
python eval.py --model-name ../models/ams-bow-stem.joblib --dataset-path ../data/beir
python eval.py --model-name ../models/ams-tf-idf-stem.joblib --dataset-path ../data/beir

python eval.py --model-name ../models/ams-all-mpnet-base-v2-stem --dataset-path ../data/beir --stemming
python eval.py --model-name ../models/ams-multi-qa-distilbert-cos-v1-stem --dataset-path ../data/beir --stemming
python eval.py --model-name ../models/ams-multi-qa-MiniLM-L6-cos-v1-stem --dataset-path ../data/beir --stemming
python eval.py --model-name ../models/ams-msmarco-distilbert-cos-v5-stem --dataset-path ../data/beir --stemming
python eval.py --model-name ../models/ams-msmarco-MiniLM-L6-cos-v5-stem --dataset-path ../data/beir --stemming
python eval.py --model-name ../models/ams-bow-stem.joblib --dataset-path ../data/beir --stemming
python eval.py --model-name ../models/ams-tf-idf-stem.joblib --dataset-path ../data/beir --stemming
