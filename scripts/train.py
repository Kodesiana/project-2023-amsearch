import os
import json
import logging
import argparse

import joblib
import pandas as pd

from gensim import utils
from gensim.models.fasttext import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import (
    InputExample,
    LoggingHandler,
    SentenceTransformer,
)

from stemmer import Stemmer, AMSTokenizer, tokenize

# ----------------------- HELPERS -----------------------


def stem_sentence(stemmer: Stemmer, text: str) -> str:
    return " ".join([stemmer.stem_ams(t) for t in tokenize(text)])


class TripletDataset(Dataset):
    def __init__(self, dataset_path: str, stemmer: Stemmer = None):
        self.stemmer = stemmer
        with open(dataset_path, "r") as f:
            self.dataset = [json.loads(x) for x in f]

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if self.stemmer:
            return InputExample(
                texts=[
                    stem_sentence(self.stemmer, item["query"]),
                    stem_sentence(self.stemmer, item["positive"]),
                    stem_sentence(self.stemmer, item["negative"]),
                ]
            )

        return InputExample(texts=[item["query"], item["positive"], item["negative"]])

    def __len__(self):
        return len(self.dataset)


class GensimTripletCorpusLoader:
    def __init__(self, corpus_path: str, stemmer: Stemmer = None):
        self.corpus_path = corpus_path
        self.stemmer = stemmer
        self.index = 0

    def __iter__(self):
        df_corpus = pd.read_json(self.corpus_path, lines=True)
        self.corpus = df_corpus.values.ravel().tolist()
        self.index = 0
        self.total = len(self.corpus)

        return self

    def __next__(self):
        if self.total == self.index:
            raise StopIteration

        line = self.corpus[self.index]
        if self.stemmer:
            line = stem_sentence(self.stemmer, line)

        self.index += 1
        return utils.simple_preprocess(line)


# ----------------------- ENTRY POINT -----------------------


# https://github.com/UKPLab/sentence-transformers/blob/master/examples/sentence_transformer/training/ms_marco/train_bi-encoder_mnrl.py
def train_bert(args):
    # load dataset
    if args.stemming:
        stemmer = Stemmer(args.vocab_path)
        train_dataset = TripletDataset(args.dataset_path, stemmer=stemmer)
    else:
        train_dataset = TripletDataset(args.dataset_path)

    # create data loader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size
    )

    # create model
    model = SentenceTransformer(args.model_name)
    model.max_seq_length = args.max_seq_length

    # create loss
    train_loss = MultipleNegativesRankingLoss(model=model)

    # train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        use_amp=True,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        # checkpoint_path=args.model_output_path,
        # checkpoint_save_steps=len(train_dataloader),
        optimizer_params={"lr": args.lr},
    )

    # save the model
    model.save(args.output_path)


def train_sklearn(args):
    # load dataset
    df_corpus = pd.read_json(args.dataset_path, lines=True)
    train_corpus = df_corpus.values.ravel().tolist()

    # create tokenizer
    tokenizer = AMSTokenizer(args.vocab_path) if args.stemming else None

    # fit model
    model = (
        TfidfVectorizer(tokenizer=tokenizer)
        if "tfidf" in args.model_name
        else CountVectorizer(tokenizer=tokenizer)
    )
    model.fit(train_corpus)

    # save model
    joblib.dump(model, args.output_path)


def train_gensim(args):
    # load dataset
    if args.stemming:
        stemmer = Stemmer(args.vocab_path)
        corpus_loader = GensimTripletCorpusLoader(args.dataset_path, stemmer=stemmer)
    else:
        corpus_loader = GensimTripletCorpusLoader(args.dataset_path)

    # create model and data
    if "doc2vec" in args.model_name:
        train_corpus = [TaggedDocument(x, [i]) for i, x in enumerate(corpus_loader)]
        gensim_model = Doc2Vec(vector_size=100, min_count=2, epochs=args.epochs)
    else:
        train_corpus = list(corpus_loader)
        gensim_model = FastText(vector_size=100, min_count=2, epochs=args.epochs)

    # build vocab and train
    gensim_model.build_vocab(train_corpus)
    gensim_model.train(
        train_corpus,
        total_examples=gensim_model.corpus_count,
        epochs=gensim_model.epochs,
    )

    # save model
    gensim_model.save(args.output_path)


def main(args):
    if "tfidf" in args.model_name or "bow" in args.model_name:
        train_sklearn(args)
        return

    if "doc2vec" in args.model_name or "fasttext" in args.model_name:
        train_gensim(args)
        return

    train_bert(args)


if __name__ == "__main__":
    #### Just some code to print debug information to stdout
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)

    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--max-seq-length", default=512, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--warmup-steps", default=1000, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
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
