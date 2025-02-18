from datetime import date
from typing import Optional
from typing_extensions import Annotated

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

STR_PK_COL = Annotated[str, mapped_column(primary_key=True)]


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id: Mapped[STR_PK_COL]
    username: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    hashed_password: Mapped[str]


class Document(db.Model):
    __tablename__ = "documents"

    id: Mapped[STR_PK_COL]
    title: Mapped[str]
    word_count: Mapped[int]
    source_url: Mapped[Optional[str]]
    published_at: Mapped[date]

    raw: Mapped["DocumentRaw"] = relationship(back_populates="parent")
    stem: Mapped["DocumentStem"] = relationship(back_populates="parent")


class DocumentRaw(db.Model):
    __tablename__ = "documents_raw"

    id: Mapped[STR_PK_COL]
    parent_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    parent: Mapped[Document] = relationship(back_populates="raw")

    title: Mapped[str]
    content: Mapped[str]
    embedding = mapped_column(Vector(768), nullable=False)


class DocumentStem(db.Model):
    __tablename__ = "documents_stem"

    id: Mapped[STR_PK_COL]
    parent_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    parent: Mapped[Document] = relationship(back_populates="stem")

    title: Mapped[str]
    content: Mapped[str]
    embedding = mapped_column(Vector(768), nullable=False)
