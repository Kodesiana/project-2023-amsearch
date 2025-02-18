from dataclasses import dataclass

from sqlalchemy import column, select, func, text
from flask_login import login_required
from flask import Blueprint, render_template

from amsearch.db import Document, DocumentRaw, db


@dataclass
class Word:
    text: str
    count: int


router = Blueprint("statistics", __name__)


@router.route("/statistics")
@login_required
def statistics():
    # Execute the first query
    stats_sums = db.session.execute(
        select(
            func.coalesce(func.count(Document.id), 0).label("total_documents"),
            func.coalesce(func.sum(Document.word_count), 0).label("total_tokens"),
        )
    ).one()

    # Check if the query returned any results
    total_tokens = stats_sums.total_tokens
    total_documents = stats_sums.total_documents

    # Safely calculate average tokens per document
    avg_tokens_per_doc = total_tokens / total_documents if total_documents > 0 else 0

    # Execute the second query
    col_text = text("unnest(string_to_array(content, ' ')) AS word")
    col_counts = func.count(DocumentRaw.id).label("counts")
    top_words_rows = db.session.execute(
        select(col_text, col_counts)
        .group_by(column("word"))
        .order_by(col_counts.desc())
        .limit(25)
    ).all()

    # Render the template with the results
    return render_template(
        "pages/admin/statistics.html",
        total_docs=f"{total_documents:,}".replace(",", "."),
        total_tokens=f"{total_tokens:,}".replace(",", "."),
        average_tokens_per_doc=f"{avg_tokens_per_doc:.2f}".replace(".", ","),
        top_words=[Word(x[0], x[1]) for x in top_words_rows],
    )
