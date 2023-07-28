import os
import uuid
from datetime import date

import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from amsearch.embeddings import Embeddings, Stemmer

class Base(DeclarativeBase):
    pass

class Document(Base):
    id: Mapped[str] = mapped_column(primary_key=True)
    title: Mapped[str]
    content: Mapped[str]
    source_url: Mapped[str]
    token_count: Mapped[int]
    published_at: Mapped[date]
    embedding_bert: Mapped[str]
    embedding_tfidf: Mapped[str]

    __tablename__ = "documents"

ROOT_MODELS = "/home/fahmi/freelance/project-2023-amsearch/data"
ROOT_DATASET = "/home/fahmi/freelance/project-2023-amsearch/dataset"

if __name__ == "__main__":
    # load embedding models
    embedding_service = Embeddings()
    embedding_service.load(ROOT_MODELS)

    # create stemmer
    stemming_service = Stemmer()
    stemming_service.load(ROOT_MODELS)

    # create postgres engine
    engine = create_engine("postgresql+psycopg2://kucingmenangis:i32F8H4kqY@localhost:5432/amsearch", echo=True)

    # read dataset
    df = pd.read_csv(f"{ROOT_DATASET}/metadata.csv", delimiter=";")
    print("Total records:", len(df))

    # proces all rows
    missing_files = []
    with Session(engine) as session:
        for index, row in df.iterrows():
            print("Inserting record", index, end="\r")

            # check file exists
            file_path = f"{ROOT_DATASET}/{row['file_name']}"
            if not os.path.isfile(file_path):
                missing_files.append(file_path)
                continue

            # read original file
            content = open(file_path, "r").read()

            # stem
            stemmed, token_count = stemming_service.stem_sentence(content)

            # make embeddings
            eb = embedding_service.extract_bert(stemmed).tolist()
            et = embedding_service.extract_tfidf(stemmed).tolist()
            
            # make document
            document = Document(
                id=str(uuid.uuid4()),
                title=row["title"],
                content=stemmed,
                source_url=row["url"],
                token_count=token_count,
                published_at=row["published_at"],
                embedding_bert=eb,
                embedding_tfidf=et
            )

            # insert
            session.add(document)
            session.commit()
    
    # print missing files
    print("Missing files:", len(missing_files))
    for file_path in missing_files:
        print(file_path)
        
    # close engine
    engine.dispose()

    print("Done")
    