import uuid
import argparse
from datetime import date
from typing_extensions import Annotated

import tqdm
import pandas as pd

from sqlalchemy import ForeignKey, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from sentence_transformers import SentenceTransformer

from stemmer import Stemmer, tokenize, word_count

# ========================================================
# DATABASE SCHEMAS
# ========================================================

STR_PK_COL = Annotated[str, mapped_column(primary_key=True)]


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[STR_PK_COL]
    title: Mapped[str]
    word_count: Mapped[int]
    source_url: Mapped[str]
    published_at: Mapped[date]

    raw: Mapped["DocumentRaw"] = relationship(back_populates="parent")
    stem: Mapped["DocumentStem"] = relationship(back_populates="parent")


class DocumentRaw(Base):
    __tablename__ = "documents_raw"

    id: Mapped[STR_PK_COL]
    parent_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    parent: Mapped[Document] = relationship(back_populates="raw")

    title: Mapped[str]
    content: Mapped[str]
    embedding = mapped_column(Vector(768), nullable=False)


class DocumentStem(Base):
    __tablename__ = "documents_stem"

    id: Mapped[STR_PK_COL]
    parent_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    parent: Mapped[Document] = relationship(back_populates="stem")

    title: Mapped[str]
    content: Mapped[str]
    embedding = mapped_column(Vector(768), nullable=False)


# ========================================================
# ENTRY POINT
# ========================================================


def main(args):
    # create postgres engine
    engine = create_engine(args.database_url, echo=True)

    # create stemmer
    stemmer = Stemmer(args.vocab_path)

    # create model
    model = SentenceTransformer(args.model_path)

    # read dataset
    df = pd.read_csv(args.dataset_path)
    print("Total records:", df.shape[0])

    # proces all rows
    with Session(engine) as session:
        # process each rows
        for index, row in (pbar := tqdm.tqdm(df.iterrows(), total=df.shape[0])):
            # report progress
            pbar.set_description(row["title"][:20].ljust(20))

            # stem document
            doc_original = row["content"]
            doc_stemmed = " ".join(map(stemmer.stem_ams, tokenize(doc_original)))

            # make embeddings
            embedding_data = model.encode([doc_original, doc_stemmed])

            # make document
            document = Document(
                # metadata
                id=str(uuid.uuid4()),
                title=row["title"],
                word_count=word_count(doc_original),
                source_url=row["source_url"],
                published_at=row["published_at"],
                # raw content
                raw=DocumentRaw(
                    id=str(uuid.uuid4()),
                    title=row["title"],
                    content=doc_original,
                    embedding=embedding_data[0, :].tolist(),
                ),
                # stemmed content
                stem=DocumentStem(
                    id=str(uuid.uuid4()),
                    title=row["title"],
                    content=doc_stemmed,
                    embedding=embedding_data[1, :].tolist(),
                ),
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
