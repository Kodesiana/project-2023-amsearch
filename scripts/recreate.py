import argparse
from datetime import date
from typing_extensions import Annotated

from sqlalchemy import ForeignKey, create_engine, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from sentence_transformers import SentenceTransformer

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

    # create model
    model = SentenceTransformer(args.model_path)

    # proces all rows
    with Session(engine) as session:
        # fetch all rows
        print("Fetching rows...")
        documents_raw = session.scalars(select(DocumentRaw)).all()
        documents_stem = session.scalars(select(DocumentStem)).all()

        # process each rows
        print("Embedding raw...")
        embedding_raw = model.encode(
            [row.content for row in documents_raw], show_progress_bar=True
        )
        print("Embedding stem...")
        embedding_stem = model.encode(
            [row.content for row in documents_stem], show_progress_bar=True
        )

        # bulk update
        print("Embedding updating...")
        session.execute(
            update(DocumentRaw),
            [
                {"id": row.id, "embedding": embedding}
                for row, embedding in zip(documents_raw, embedding_raw.tolist())
            ],
        )
        session.execute(
            update(DocumentStem),
            [
                {"id": row.id, "embedding": embedding}
                for row, embedding in zip(documents_stem, embedding_stem.tolist())
            ],
        )

        # bulk commit
        print("Commiting changes...")
        session.commit()

    # close engine
    engine.dispose()

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--vocab-path", type=str, required=True)
    parser.add_argument("--database-url", type=str, required=True)

    args = parser.parse_args()
    print(args)

    main(args)
