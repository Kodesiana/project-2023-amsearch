import time
from dataclasses import dataclass

import requests
from sqlalchemy import func

from amsearch.db import db, Document
from amsearch.embeddings import Embeddings


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


EMPTY_RESULT = Results(0, 0, 0, [], False, False)
LIMIT_DISTANCE = 1

class SearchService:

    def __init__(self, cse_id: str, cse_api_key: str, embedding: Embeddings):
        self.cse_id = cse_id
        self.cse_api_key = cse_api_key
        self.embedding = embedding

    def search(self,
               engine: str,
               q: str,
               page: int = 0,
               per_page: int = 10) -> Results:
        # run search
        if engine == "google":
            return self.__google_search(q, page, per_page)
        elif engine == "tfidf":
            return self.__tfidf_search(q, page, per_page)
        elif engine == "bert":
            return self.__bert_search(q, page, per_page)

    def __truncate_contents(self, contents: str, max_length: int = 200) -> str:
        if len(contents) > max_length:
            return contents[:max_length] + "..."
        else:
            return contents

    def __google_search(self, q: str, page: int, limit: int) -> Results:
        # google has custom paging offset
        start = (page - 1) * limit + 1

        # build URL
        url = f"https://www.googleapis.com/customsearch/v1?key={self.cse_api_key}&cx={self.cse_id}&q={q}&start={start}&num={limit}"

        # run query
        response = requests.get(url)
        response.raise_for_status()

        # build results
        body = response.json()

        has_prev = "previousPage" in body["queries"]
        has_next = "nextPage" in body["queries"]
        results = [
            ResultItem(
                title=result["title"],
                url=result["link"],
                excerpt=result["snippet"],
                distance=0,
            ) for result in body["items"]
        ]

        return Results(
            execution_time=body["searchInformation"]["searchTime"],
            total=int(body["searchInformation"]["totalResults"]),
            page=page,
            items=results,
            has_prev=has_prev,
            has_next=has_next,
        )
    
    def ___embedding_search(self, q: str, page: int, per_page: int, engine: str) -> Results:
        # extract embeddings
        embedding = []
        distance_col = None
        if engine == "tfidf":
            embedding = self.embedding.extract_tfidf(q)
            distance_col = Document.embedding_tfidf.cosine_distance(embedding).label("distance")
        else:
            embedding = self.embedding.extract_bert(q)
            distance_col = Document.embedding_bert.cosine_distance(embedding).label("distance")

        # build query
        rows_query = db.select(Document.title, Document.source_url, Document.content, distance_col) \
            .where(distance_col < LIMIT_DISTANCE) \
            .order_by(distance_col) \
            .limit(per_page) \
            .offset((page - 1) * per_page)
        total_query = db.select(func.count()) \
            .select_from(Document) \
            .where(distance_col < LIMIT_DISTANCE)

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
            results.append(ResultItem(
                title=result[0],
                url=result[1],
                excerpt=self.__truncate_contents(result[2]),
                distance=result[3]
            ))

        return Results(execution_time=execution_time,
                       total=total_count,
                       page=page,
                       items=results,
                       has_prev=page > 1,
                       has_next=page < total_pages)

    def __tfidf_search(self, q: str, page: int, per_page: int) -> Results:
        return self.___embedding_search(q, page, per_page, "tfidf")

    def __bert_search(self, q: str, page: int, per_page: int) -> Results:
        return self.___embedding_search(q, page, per_page, "bert")
