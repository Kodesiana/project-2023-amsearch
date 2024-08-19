from dataclasses import dataclass

from flask import Blueprint, render_template, flash, request

from amsearch.services import VectorSearchInstance


@dataclass
class StemResult:
    original: str
    ams: str
    purwoko: str
    sastrawi: str
    ug18: str


router = Blueprint("stemming", __name__)


@router.route("/stemming", methods=["GET", "POST"])
def stem():
    # show page
    if request.method == "GET":
        return render_template(
            "pages/public/stemming.html", input_text="", output_text="", stems=[]
        )

    # get text
    input_text = request.form.get("input-text")
    if not input_text:
        return render_template(
            "pages/public/stemming.html", input_text="", output_text="", stems=[]
        )

    # split tokens
    tokens = set(
        [word.strip().lower() for word in VectorSearchInstance.tokenize(input_text)]
    )
    stems = [
        StemResult(
            original=word,
            ams=VectorSearchInstance.stemmer.stem_ams(word),
            purwoko=VectorSearchInstance.stemmer.stem_purwoko(word),
            sastrawi=VectorSearchInstance.stemmer.stem_sastrawi(word),
            ug18=VectorSearchInstance.stemmer.stem_ug18(word),
        )
        for word in tokens
    ]

    return render_template(
        "pages/public/stemming.html",
        input_text=input_text,
        output_ams=VectorSearchInstance.stem_sentence(input_text, "ams")[0],
        output_purwoko=VectorSearchInstance.stem_sentence(input_text, "purwoko")[0],
        output_sastrawi=VectorSearchInstance.stem_sentence(input_text, "sastrawi")[0],
        output_ug18=VectorSearchInstance.stem_sentence(input_text, "ug18")[0],
        stems=stems,
    )
