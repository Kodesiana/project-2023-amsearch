from dataclasses import dataclass

from sqlalchemy import text

from flask import Blueprint, render_template, flash, request
from flask_login import login_required

from amsearch.services import VectorSearchInstance
from amsearch.db import db


@dataclass
class Word:
    text: str
    count: int


router = Blueprint("statistics", __name__)


@router.route("/statistics")
@login_required
def statistics():
    statistic_sums = db.session.execute(
        text(
            "SELECT COUNT(*) AS total_documents, SUM(token_count) AS total_tokens FROM documents"
        )
    ).one()
    top_words_rows = db.session.execute(
        text(
            "SELECT unnest(string_to_array(content, ' ')) AS text, count(*) AS count FROM documents GROUP BY 1 ORDER BY 2 DESC LIMIT 25"
        )
    )

    avg_tokens_per_doc = statistic_sums[1] / statistic_sums[0]

    return render_template(
        "pages/admin/statistics.html",
        total_docs=f"{statistic_sums[0]:,}".replace(",", "."),
        total_tokens=f"{statistic_sums[1]:,}".replace(",", "."),
        average_tokens_per_doc=f"{avg_tokens_per_doc:.2f}",
        top_words=[Word(x[0], x[1]) for x in top_words_rows],
    )
