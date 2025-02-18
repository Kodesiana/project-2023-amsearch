import time
from dataclasses import dataclass

from thefuzz import process
from sqlalchemy import func, select
from sentence_transformers import SentenceTransformer

from amsearch.db import db, Document, DocumentStem
from amsearch.stemmer import Stemmer, tokenize


@dataclass
class ResultItem:
    title: str
    url: str
    excerpt: str
    distance: float

    def __repr__(self):
        return f"<ResultItem title={self.title}>"


@dataclass
class Results:
    execution_time: int
    total: int
    page: int
    items: list[ResultItem]
    has_prev: bool
    has_next: bool


LIMIT_DISTANCE = 1.1
EMPTY_RESULT = Results(0, 0, 0, [], False, False)


class SearchService:
    def load(self, vocab_path: str, embedding_model_path: str):
        self.stemmer = Stemmer(vocab_path)
        self.model = SentenceTransformer.load(embedding_model_path)

    def search(
        self, stemmer: str, q: str, page: int = 0, per_page: int = 10
    ) -> Results:
        # perform stemming
        stemmed = self.stem_sentence(q, stemmer)

        # extract embeddings
        embedding = self.model.encode([stemmed])[0]
        distance_col = DocumentStem.embedding.cosine_distance(embedding).label(
            "distance"
        )

        # build query
        rows_query = (
            select(
                DocumentStem.title,
                DocumentStem.content,
                distance_col,
                Document.source_url,
            )
            .join(DocumentStem.parent)
            .where(distance_col < LIMIT_DISTANCE)
            .order_by(distance_col)
            .limit(per_page)
            .offset((page - 1) * per_page)
        )
        total_query = (
            select(func.count())
            .select_from(DocumentStem)
            .where(distance_col < LIMIT_DISTANCE)
        )

        # run query
        start_time = time.time()
        paged_rows = db.session.execute(rows_query).all()
        total_count = db.session.execute(total_query).scalar()
        execution_time = time.time() - start_time

        # calculate pages
        total_pages = total_count // per_page + (1 if total_count % per_page > 0 else 0)

        # build results
        results = []
        for result in paged_rows:
            results.append(
                ResultItem(
                    title=result[0],
                    excerpt=self.__truncate_contents(result[1]),
                    distance=result[2],
                    url=result[3],
                )
            )

        return Results(
            execution_time=execution_time,
            total=total_count,
            page=page,
            items=results,
            has_prev=page > 1,
            has_next=page < total_pages,
        )

    def embed(self, text: str):
        return self.model.encode([text])[0]

    def stem_sentence(self, sentence: str, stemmer: str) -> str:
        stems = tokenize(sentence)
        if stemmer == "ams":
            stems = list(map(self.stemmer.stem_ams, stems))
        elif stemmer == "purwoko":
            stems = list(map(self.stemmer.stem_purwoko, stems))
        elif stemmer == "sastrawi":
            stems = list(map(self.stemmer.stem_sastrawi, stems))
        elif stemmer == "ug18":
            stems = list(map(self.stemmer.stem_ug18, stems))

        return " ".join(stems)

    def extract_fuzzy_alternatives(self, *args, **kwargs):
        return process.extract(*args, **kwargs)

    def __truncate_contents(self, contents: str, max_length: int = 200) -> str:
        if len(contents) > max_length:
            return contents[:max_length] + "..."
        else:
            return contents


IR = SearchService()
