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
  # Execute the first query
  try:
    statistic_sums = db.session.execute(
      text(
        "SELECT COUNT(*) AS total_documents, SUM(token_count) AS total_tokens FROM documents"
      )
    ).one()
  except Exception as e:
    flash(f"Gagal mengakses statistik: {str(e)}", "danger")
    return render_template("pages/admin/statistics.html")

  # Check if the query returned any results
  total_documents = statistic_sums[0] if statistic_sums[0] else 0
  total_tokens = statistic_sums[1] if statistic_sums[1] else 0

  # Safely calculate average tokens per document
  avg_tokens_per_doc = total_tokens / total_documents if total_documents > 0 else 0

  # Execute the second query
  try:
    top_words_rows = db.session.execute(
      text(
        "SELECT unnest(string_to_array(content, ' ')) AS text, count(*) AS count FROM documents GROUP BY 1 ORDER BY 2 DESC LIMIT 25"
      )
    )
  except Exception as e:
    flash(f"Gagal mengambil data statistik: {str(e)}", "danger")
    top_words_rows = []

  # Render the template with the results
  return render_template(
    "pages/admin/statistics.html",
    total_docs=f"{total_documents:,}".replace(",", "."),
    total_tokens=f"{total_tokens:,}".replace(",", "."),
    average_tokens_per_doc=f"{avg_tokens_per_doc:.2f}",
    top_words=[Word(x[0], x[1]) for x in top_words_rows],
  )
