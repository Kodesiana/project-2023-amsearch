from flask import Blueprint, render_template, flash, request

from amsearch.services import EMPTY_RESULT, VectorSearchInstance

router = Blueprint("search", __name__)

AVAILABLE_STEMMER = set(["none", "ams", "purwoko", "sastrawi", "ug18"])
STEMMER_DESC = {
    "none": "Tanpa stemming",
    "ams": "AMS",
    "purwoko": "Purwoko",
    "sastrawi": "Sastrawi",
    "ug18": "UG18",
}


@router.route("/search")
def search():
    # get search engine
    stemmer = request.args.get("stemmer", "none")
    if stemmer not in AVAILABLE_STEMMER:
        stemmer = "none"

    # get keyword
    q = request.args.get("q", "")
    if not q:
        return render_template(
            "pages/public/search.html",
            stemmer=stemmer,
            keyword="",
            execution_time="0.0",
            results=EMPTY_RESULT,
        )

    # get page and per_page
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)

    # run search
    results = VectorSearchInstance.search(stemmer, q, page, per_page)

    return render_template(
        "pages/public/search.html",
        stemmer=STEMMER_DESC[stemmer],
        keyword=q,
        execution_time=f"{results.execution_time:.2f}",
        results=results,
    )
