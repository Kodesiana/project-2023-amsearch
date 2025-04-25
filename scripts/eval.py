import os
import logging
import datetime
import argparse

import joblib
import numpy as np
import pandas as pd

from beir import LoggingHandler
from beir.retrieval.models import SentenceBERT
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch

from stemmer import Stemmer, AMSTokenizer, tokenize

# ----------------------- HELPERS -----------------------

K_VALUES = [1, 3, 5, 10, 100, 1000]


def save_results(model_name: str, eval_stemming: bool, items: list[dict[str, float]]):
    metrics = [
        {
            "model": model_name,
            "stemming_model": "stem" in model_name,
            "stemming_eval": eval_stemming,
            "metric": k.split("@")[0],
            "k": k.split("@")[1],
            "value": v,
        }
        for col in items
        for k, v in col.items()
    ]

    safe_model_name = model_name.split("/")[-1]
    safe_model_name = safe_model_name.replace(".joblib", "")
    safe_model_name = (
        f"{safe_model_name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(safe_model_name, index=None)


def stem_sentence(stemmer: Stemmer, text: str) -> str:
    return " ".join([stemmer.stem_ams(t) for t in tokenize(text)])


def stem_corpus(stemmer: Stemmer, item: dict[str, str]):
    return {k: stem_sentence(stemmer, v) for k, v in item.items()}


# https://github.com/beir-cellar/beir/wiki/Evaluate-your-custom-model
class AMSBaselineModel:
    def __init__(self, model_name: str, vocab_path: str, **kwargs):
        self.model = joblib.load(model_name)

        if "stem" in model_name:
            self.model.tokenizer = AMSTokenizer(vocab_path)

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(
        self, queries: list[str], batch_size: int, **kwargs
    ) -> np.ndarray:
        return self.model.transform(queries).todense().astype(float)

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(
        self, corpus: list[dict[str, str]], batch_size: int, **kwargs
    ) -> np.ndarray:
        extracted_corpus = [row["title"] + " " + row["text"] for row in corpus]
        return self.model.transform(extracted_corpus).todense().astype(float)


# ----------------------- ENTRY POINT -----------------------


def load_model(args):
    if "tf-idf" in args.model_name or "bow" in args.model_name:
        return DenseRetrievalExactSearch(
            AMSBaselineModel(args.model_name, args.vocab_path),
            batch_size=args.batch_size,
        )

    return DenseRetrievalExactSearch(
        SentenceBERT(args.model_name), batch_size=args.batch_size
    )


def main(args):
    # load dataset
    corpus, queries, qrels = GenericDataLoader(
        data_folder=args.dataset_path, qrels_file=f"{args.dataset_path}/qrels.tsv"
    ).load_custom()

    # optionally, stemming
    if args.stemming:
        stemmer = Stemmer(args.vocab_path)
        queries = stem_corpus(stemmer, queries)
        corpus = {k: stem_corpus(stemmer, v) for k, v in corpus.items()}

    # load the model
    model = load_model(args)

    # retrieve data
    retriever = EvaluateRetrieval(model, score_function="cos_sim")
    results = retriever.retrieve(corpus, queries)

    # evaluate model with NDCG@k, MAP@K, Recall@K and Precision@K
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, K_VALUES)

    # save metrics
    save_results(args.model_name, args.stemming, [ndcg, _map, recall, precision])


if __name__ == "__main__":
    # print debug information to stdout
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    # create CLI parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stemming", action="store_true")
    parser.add_argument(
        "--vocab-path", type=str, default="../data/sundabaru1-vocab.txt"
    )

    args = parser.parse_args()
    print(args)

    if args.stemming:
        if args.vocab_path is None or not os.path.exists(args.vocab_path):
            print("VOCAB NOT FOUND")
            exit(-1)

    main(args)
