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
    title = db.Column(db.String(256), unique=False, nullable=True)
    content = db.Column(db.Text, unique=False, nullable=True)
    source_url = db.Column(db.Text, unique=False, nullable=True)
    token_count = db.Column(db.Integer, unique=False, nullable=True)
    published_at = db.Column(db.Date, primary_key=False, nullable=True)
    embedding_bert = db.Column(Vector(768))
    embedding_tfidf = db.Column(Vector(768))

    __tablename__ = "documents"

    def __repr__(self):
        return f"<Document {self.title}>"
