import uuid
import argparse
from datetime import date

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sentence_transformers import SentenceTransformer

from amsearch.stemmer import Stemmer, tokenize

# ========================================================
# DATABASE SCHEMAS
# ========================================================


class Base(DeclarativeBase):
    pass


class Document(Base):
    id: Mapped[str] = mapped_column(primary_key=True)
    title: Mapped[str]

    source: Mapped[str]
    word_count: Mapped[int]
    published_at: Mapped[date]

    content_raw: Mapped[str]
    content_ams: Mapped[str]
    embedding_raw: Mapped[str]
    embedding_ams: Mapped[str]

    __tablename__ = "documents"


# ========================================================
# ENTRY POINT
# ========================================================


def main(args):
    # create stemmer
    stemmer = Stemmer(args.vocab_path)

    # create model
    model = SentenceTransformer(args.model_path)

    # create postgres engine
    engine = create_engine(args.database_url, echo=True)

    # read dataset
    df = pd.read_json(model, lines=True)
    print("Total records:", df.shape[0])

    # proces all rows
    with Session(engine) as session:
        # process each rows
        for index, row in df.iterrows():
            print("Inserting record", index, end="\r")

            # stem document
            stemmed, word_count = [
                stemmer.stem_ams(word.strip().lower())
                for word in tokenize(row["content"])
            ]

            # make embeddings
            embedding_data = model.encode(
                [
                    row["content"],
                    stemmed,
                ]
            )

            # make document
            document = Document(
                id=str(uuid.uuid4()),
                title=row["title"],
                source=row["url"],
                word_count=word_count,
                published_at=row["published_at"],
                content_raw=row["content"],
                content_ams=stemmed,
                embedding_raw=embedding_data[0, :].tolist(),
                embedding_ams=embedding_data[1, :].tolist(),
            )

            # insert
            session.add(document)

        # bulk commit
        print("Commiting changes...")
        session.commit()

    # close engine
    engine.dispose()

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--database_url", type=str, required=True)

    args = parser.parse_args()
    print(args)

    main(args)
