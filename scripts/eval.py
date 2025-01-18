import logging
import argparse

import bm25s
from beir import LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch

from amsearch.services import Stemmer


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


def eval_bm25(args):
    # load dataset
    corpus, queries, qrels = GenericDataLoader(
        data_folder=args.dataset_path, qrels_file=f"{args.dataset_path}/qrels.tsv"
    ).load_custom()

    corpus_ids, corpus_lst = [], []
    for key, val in corpus.items():
        corpus_ids.append(key)
        corpus_lst.append(val["title"] + " " + val["text"])

    qids, queries_lst = [], []
    for key, val in queries.items():
        qids.append(key)
        queries_lst.append(val)

    if args.model_name == "bm25-ams":
        stemmer = Stemmer(args.vocab_path)
        corpus_tokens = bm25s.tokenize(
            corpus_lst, stemmer=lambda lst: list(map(stemmer.stem_ams, lst))
        )
        query_tokens = bm25s.tokenize(
            queries_lst, stemmer=lambda lst: list(map(stemmer.stem_ams, lst))
        )
    else:
        corpus_tokens = bm25s.tokenize(corpus_lst)
        query_tokens = bm25s.tokenize(queries_lst)

    model = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    model.index(corpus_tokens)

    queried_results, queried_scores = model.retrieve(
        query_tokens, corpus=corpus_ids, k=1000, n_threads=1
    )
    results_dict = postprocess_results_for_eval(queried_results, queried_scores, qids)

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results_dict, [1, 3, 5, 10, 100, 1000]
    )


def eval_dense(args):
    # load dataset
    corpus, queries, qrels = GenericDataLoader(
        data_folder=args.dataset_path, qrels_file=f"{args.dataset_path}/qrels.tsv"
    ).load_custom()

    # load the SBERT model
    model = DenseRetrievalExactSearch(
        models.SentenceBERT(args.model_name), batch_size=args.batch_size
    )

    # retrieve
    retriever = EvaluateRetrieval(model, score_function=args.score_function)
    results = retriever.retrieve(corpus, queries)

    # evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results, retriever.k_values
    )


def main(args):
    if args.model_name == "bm25" or args.model_name == "bm25-ams":
        eval_bm25(args)
    else:
        eval_dense(args)


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
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--score_function", type=str, default="dot", choices=["dot", "cos_sim"]
    )
    parser.add_argument("--vocab_path", type=str)

    args = parser.parse_args()
    print(args)

    # run app
    main(args)
