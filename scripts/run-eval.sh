#!/usr/bin/env bash
set -euxo pipefail

# --- eval pretrained models
python eval.py --model-name sentence-transformers/all-mpnet-base-v2 --dataset-path ../data/dataset-new/cleaned 
python eval.py --model-name sentence-transformers/multi-qa-distilbert-cos-v1 --dataset-path ../data/dataset-new/cleaned 
python eval.py --model-name sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --dataset-path ../data/dataset-new/cleaned 
python eval.py --model-name sentence-transformers/msmarco-distilbert-cos-v5 --dataset-path ../data/dataset-new/cleaned 
python eval.py --model-name sentence-transformers/msmarco-MiniLM-L6-cos-v5 --dataset-path ../data/dataset-new/cleaned 
python eval.py --model-name StevenLimcorn/MelayuBERT --dataset-path ../data/dataset-new/cleaned 
python eval.py --model-name indolem/indobert-base-uncased --dataset-path ../data/dataset-new/cleaned

python eval.py --model-name sentence-transformers/all-mpnet-base-v2 --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name sentence-transformers/multi-qa-distilbert-cos-v1 --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name sentence-transformers/msmarco-distilbert-cos-v5 --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name sentence-transformers/msmarco-MiniLM-L6-cos-v5 --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name StevenLimcorn/MelayuBERT --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name indolem/indobert-base-uncased --dataset-path ../data/dataset-new/cleaned --stemming

# --- eval tuned models
# non stemming
python eval.py --model-name ../models/ams-all-mpnet-base-v2 --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-multi-qa-distilbert-cos-v1 --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-multi-qa-MiniLM-L6-cos-v1 --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-msmarco-distilbert-cos-v5 --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-msmarco-MiniLM-L6-cos-v5 --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-melayubert --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-indobert --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-bow.joblib --dataset-path ../data/dataset-new/cleaned 
python eval.py --model-name ../models/ams-tfidf.joblib --dataset-path ../data/dataset-new/cleaned 
python eval.py --model-name ../models/ams-doc2vec.gensim --dataset-path ../data/dataset-new/cleaned 
python eval.py --model-name ../models/ams-fasttext.gensim --dataset-path ../data/dataset-new/cleaned 

python eval.py --model-name ../models/ams-all-mpnet-base-v2 --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-multi-qa-distilbert-cos-v1 --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-multi-qa-MiniLM-L6-cos-v1 --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-msmarco-distilbert-cos-v5 --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-msmarco-MiniLM-L6-cos-v5 --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-melayubert --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-indobert --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-bow.joblib --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-tfidf.joblib --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-doc2vec.gensim --dataset-path ../data/dataset-new/cleaned  --stemming
python eval.py --model-name ../models/ams-fasttext.gensim --dataset-path ../data/dataset-new/cleaned  --stemming

# stemming
python eval.py --model-name ../models/ams-all-mpnet-base-v2-stem --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-multi-qa-distilbert-cos-v1-stem --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-multi-qa-MiniLM-L6-cos-v1-stem --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-msmarco-distilbert-cos-v5-stem --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-msmarco-MiniLM-L6-cos-v5-stem --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-melayubert-stem --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-indobert-stem --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-bow-stem.joblib --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-tfidf-stem.joblib --dataset-path ../data/dataset-new/cleaned
python eval.py --model-name ../models/ams-doc2vec-stem.gensim --dataset-path ../data/dataset-new/cleaned 
python eval.py --model-name ../models/ams-fasttext-stem.gensim --dataset-path ../data/dataset-new/cleaned 

python eval.py --model-name ../models/ams-all-mpnet-base-v2-stem --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-multi-qa-distilbert-cos-v1-stem --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-multi-qa-MiniLM-L6-cos-v1-stem --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-msmarco-distilbert-cos-v5-stem --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-msmarco-MiniLM-L6-cos-v5-stem --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-melayubert-stem --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-indobert-stem --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-bow-stem.joblib --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-tfidf-stem.joblib --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-doc2vec-stem.gensim --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../models/ams-fasttext-stem.gensim --dataset-path ../data/dataset-new/cleaned --stemming

# perbandingan 50 dan 100 epochs
python eval.py --model-name ../tmp/ams-indobert-stem-50 --dataset-path ../data/dataset-new/cleaned --stemming
python eval.py --model-name ../tmp/ams-indobert-stem-100 --dataset-path ../data/dataset-new/cleaned --stemming
