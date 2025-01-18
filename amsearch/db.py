from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

from pgvector.sqlalchemy import Vector

db = SQLAlchemy()


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    hashed_password = db.Column(db.String(), unique=False, nullable=False)

    __tablename__ = "users"

    def __repr__(self):
        return f"<User {self.username}  {self.hashed_password}>"


class Document(db.Model):
    id = db.Column(db.String(256), primary_key=True)
    title = db.Column(db.String(256), unique=False, nullable=False)

    source = db.Column(db.Text, unique=False, nullable=True)
    word_count = db.Column(db.Integer, unique=False, nullable=False)
    published_at = db.Column(db.Date, primary_key=False, nullable=False)

    content_raw = db.Column(db.Text, unique=False, nullable=False)
    content_ams = db.Column(db.Text, unique=False, nullable=False)
    embedding_raw = db.Column(Vector(768))
    embedding_ams = db.Column(Vector(768))

    __tablename__ = "documents"

    def __repr__(self):
        return f"<Document {self.title}>"
