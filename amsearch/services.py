import time
from dataclasses import dataclass

import requests

from amsearch.db import db, Document
from amsearch.embeddings import Embeddings


@dataclass
class ResultItem:
    title: str
    url: str
    excerpt: str

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

    def __tfidf_search(self, q: str, page: int, per_page: int) -> Results:
        # extract embeddings
        embedding = self.embedding.extract_tfidf(q)

        # run query
        start_time = time.time()
        query = db.select(Document) \
            .order_by(Document.embedding_tfidf.cosine_distance(embedding))
        paged_rows = db.paginate(query, page=page, per_page=per_page)

        # build results
        execution_time = time.time() - start_time
        results = [
            ResultItem(
                title=result.title,
                url=result.source_url,
                excerpt=self.__truncate_contents(result.content),
            ) for result in paged_rows.items
        ]

        return Results(execution_time=execution_time,
                       total=paged_rows.total,
                       page=page,
                       items=results,
                       has_prev=paged_rows.has_prev,
                       has_next=paged_rows.has_next)

    def __bert_search(self, q: str, page: int, per_page: int) -> Results:
        # extract embeddings
        embedding = self.embedding.extract_bert(q)

        # run query
        start_time = time.time()
        query = db.select(Document).order_by(
            Document.embedding_bert.cosine_distance(embedding))
        paged_rows = db.paginate(query, page=page, per_page=per_page)

        # build results
        execution_time = time.time() - start_time
        results = [
            ResultItem(
                title=result.title,
                url=result.source_url,
                excerpt=self.__truncate_contents(result.content),
            ) for result in paged_rows.items
        ]

        return Results(execution_time=execution_time,
                       total=paged_rows.total,
                       page=page,
                       items=results,
                       has_prev=paged_rows.has_prev,
                       has_next=paged_rows.has_next)
