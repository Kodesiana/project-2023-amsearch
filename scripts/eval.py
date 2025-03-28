import logging
import argparse

import bm25s
import pandas as pd

from beir import LoggingHandler
from beir.retrieval.models import SentenceBERT
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch

from stemmer import Stemmer

# ----------------------- HELPERS -----------------------

K_VALUES = [1, 3, 5, 10, 100, 1000]

# https://www.kaggle.com/code/xhlulu/benchmark-bm25-on-beir
def postprocess_results_for_eval(results, scores, query_ids):
    results_record = [
        {"id": qid, "hits": results[i], "scores": list(scores[i])}
        for i, qid in enumerate(query_ids)
    ]

    result_dict_for_eval = {
        res["id"]: {
            docid: float(score) for docid, score in zip(res["hits"], res["scores"])
        }
        for res in results_record
    }

    return result_dict_for_eval

def save_results(model_name: str, items: list[dict[str, float]]):
    metrics = [{"metric": k.split("@")[0], "k": k.split("@")[1], "value": v} for col in items for k, v in col.items()]
    df_metrics = pd.DataFrame(metrics)

    safe_model_name = model_name.replace("/", "_")
    df_metrics.to_csv(f"eval-{safe_model_name}.csv", index=None)

# ----------------------- COMMANDS -----------------------

def eval_bm25(args):
    # load dataset
    corpus, queries, qrels = GenericDataLoader(data_folder=args.dataset_path, qrels_file=f"{args.dataset_path}/qrels.tsv").load_custom()

    # build BM25 corpus
    corpus_ids, corpus_lst = [], []
    for key, val in corpus.items():
        corpus_ids.append(key)
        corpus_lst.append(val["text"])
        # corpus_lst.append(val["title"] + " " + val["text"])

    qids, queries_lst = [], []
    for key, val in queries.items():
        qids.append(key)
        queries_lst.append(val)

    # tokenize data, optionally with tokenization
    if args.eval_mode == "bm25-ams":
        stemmer = Stemmer(args.vocab_path)
        corpus_tokens = bm25s.tokenize(corpus_lst, stemmer=lambda lst: list(map(stemmer.stem_ams, lst)))
        query_tokens = bm25s.tokenize(queries_lst, stemmer=lambda lst: list(map(stemmer.stem_ams, lst)))
    else:
        corpus_tokens = bm25s.tokenize(corpus_lst)
        query_tokens = bm25s.tokenize(queries_lst)

    # index data
    model = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    model.index(corpus_tokens)

    # retrieve data
    queried_results, queried_scores = model.retrieve(query_tokens, corpus=corpus_ids, k=1000, n_threads=1)
    results_dict = postprocess_results_for_eval(queried_results, queried_scores, qids)

    # evaluate model with NDCG@k, MAP@K, Recall@K and Precision@K
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results_dict, K_VALUES)

    # save metrics
    save_results(args.model_name, [ndcg, _map, recall, precision])


def eval_dense(args):
    # load dataset
    corpus, queries, qrels = GenericDataLoader(data_folder=args.dataset_path, qrels_file=f"{args.dataset_path}/qrels.tsv").load_custom()

    # load the SBERT model
    model = DenseRetrievalExactSearch(SentenceBERT(args.model_name), batch_size=args.batch_size)

    # retrieve data
    retriever = EvaluateRetrieval(model, score_function=args.score_function)
    results = retriever.retrieve(corpus, queries)

    # evaluate model with NDCG@k, MAP@K, Recall@K and Precision@K
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, K_VALUES)

    # save metrics
    save_results(args.model_name, [ndcg, _map, recall, precision])


# ----------------------- ENTRY POINT -----------------------

def main(args):
    if args.eval_mode == "dense":
        eval_dense(args)
    else:
        eval_bm25(args)


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
    parser.add_argument("--eval-mode", type=str, default="dense", choices=["dense", "bm25", "bm25-ams"])
    parser.add_argument("--vocab-path", type=str, default="data/sundabaru1-vocab.txt")
    parser.add_argument("--score-function", type=str, default="dot", choices=["dot", "cos_sim"])

    args = parser.parse_args()
    main(args)
