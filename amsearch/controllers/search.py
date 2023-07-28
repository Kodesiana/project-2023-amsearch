from flask import Blueprint, render_template, flash, request

from amsearch import search_service, stemming_service
from amsearch.services import EMPTY_RESULT

router = Blueprint('search', __name__)

AVAILABLE_ENGINES = {
    'bert': "Mesin Pencari AMSearch-BERT",
    'tfidf': "Mesin Pencari AMSearch-TF IDF",
    'google': "Mesin Pencari Google",
}


@router.route('/')
def home():
    return render_template("pages/public/home.html")


@router.route('/search')
def search():
    # get search engine
    search_engine = request.args.get('engine', 'bert')
    if search_engine not in AVAILABLE_ENGINES:
        flash("Mesin pencari tidak tersedia")
        return render_template("pages/public/search.html",
                               results=EMPTY_RESULT)

    # get keyword
    q = request.args.get('q', '')
    if not q:
        return render_template(
            "pages/public/search.html",
            engine_desc=AVAILABLE_ENGINES[search_engine],
            engine=search_engine,
            keyword="",
            execution_time="0.0",
            results=EMPTY_RESULT,
        )

    # get page and per_page
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)

    # stem input
    q, _ = stemming_service.stem_sentence(q)

    # run search
    results = search_service.search(search_engine, q, page, per_page)

    return render_template(
        "pages/public/search.html",
        engine_desc=AVAILABLE_ENGINES[search_engine],
        engine=search_engine,
        keyword=q,
        execution_time=f"{results.execution_time:.2f}",
        results=results,
    )


@router.route('/about')
def about():
    return render_template("pages/public/about.html")
