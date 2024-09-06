from dataclasses import dataclass

import numpy as np
from thefuzz import process
from flask import Blueprint, render_template, request

from amsearch.services import VectorSearchInstance


@dataclass
class StemResult:
    original: str
    ams: str
    purwoko: str
    sastrawi: str
    ug18: str


router = Blueprint("stemming", __name__)


def stats_description(stem_stats, prefix, total_unique):
    return "Jumlah kata stemming: {0}\nJumlah kata unik: {1}\nJumlah kata benar: {2}\n Akurasi: {3:.2f}%".format(
        stem_stats[f"stemmed_{prefix}"],
        total_unique,
        stem_stats[f"correct_{prefix}"],
        stem_stats[f"accuracy_{prefix}"],
    )


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

    stems: list[StemResult] = [
        StemResult(
            original=word,
            ams=VectorSearchInstance.stemmer.stem_ams(word),
            purwoko=VectorSearchInstance.stemmer.stem_purwoko(word),
            sastrawi=VectorSearchInstance.stemmer.stem_sastrawi(word),
            ug18=VectorSearchInstance.stemmer.stem_ug18(word),
        )
        for word in tokens
    ]

    stem_stats = {
        "stemmed_ams": np.sum([res.original != res.ams for res in stems]),
        "stemmed_purwoko": np.sum([res.original != res.purwoko for res in stems]),
        "stemmed_sastrawi": np.sum([res.original != res.sastrawi for res in stems]),
        "stemmed_ug18": np.sum([res.original != res.ug18 for res in stems]),
        "correct_ams": np.sum(
            [x.ams in VectorSearchInstance.stemmer.kamus for x in stems]
        ),
        "correct_purwoko": np.sum(
            [x.purwoko in VectorSearchInstance.stemmer.kamus for x in stems]
        ),
        "correct_sastrawi": np.sum(
            [x.sastrawi in VectorSearchInstance.stemmer.kamus for x in stems]
        ),
        "correct_ug18": np.sum(
            [x.ug18 in VectorSearchInstance.stemmer.kamus for x in stems]
        ),
    }

    # calculate true positives
    stem_stats = {
        **stem_stats,
        "accuracy_ams": stem_stats["correct_ams"] / len(stems) * 100,
        "accuracy_purwoko": stem_stats["correct_purwoko"] / len(stems) * 100,
        "accuracy_sastrawi": stem_stats["correct_sastrawi"] / len(stems) * 100,
        "accuracy_ug18": stem_stats["correct_ug18"] / len(stems) * 100,
    }

    return render_template(
        "pages/public/stemming.html",
        # input text
        input_text=input_text,
        # sentence stemming
        output_ams=VectorSearchInstance.stem_sentence(input_text, "ams")[0],
        output_purwoko=VectorSearchInstance.stem_sentence(input_text, "purwoko")[0],
        output_sastrawi=VectorSearchInstance.stem_sentence(input_text, "sastrawi")[0],
        output_ug18=VectorSearchInstance.stem_sentence(input_text, "ug18")[0],
        # per word stems
        stems=stems,
        # statistics
        stats_ams=stats_description(stem_stats, "ams", len(stems)),
        stats_purwoko=stats_description(stem_stats, "purwoko", len(stems)),
        stats_sastwawi=stats_description(stem_stats, "sastrawi", len(stems)),
        stats_ug18=stats_description(stem_stats, "ug18", len(stems)),
    )


@router.route("/stemming_word", methods=["GET", "POST"])
def stem_word():
    # show page
    if request.method == "GET":
        return render_template(
            "pages/public/stemming_single.html",
            input_word="",
            stemmed_word="",
            alternatives=[],
        )

    # get text
    input_word = request.form.get("input-word")
    if not input_word:
        return render_template(
            "pages/public/stemming_single.html",
            input_word="",
            stemmed_word="",
            alternatives=[],
        )

    # stem the word
    stemmed_word = VectorSearchInstance.stemmer.stem_ams(input_word.lower())

    # find alternative if the word is not in the dictionary
    alternatives = []
    if stemmed_word not in VectorSearchInstance.stemmer.kamus:
        alternatives = process.extract(
            stemmed_word, VectorSearchInstance.stemmer.kamus, limit=5
        )

    # render page
    return render_template(
        "pages/public/stemming_single.html",
        input_word=input_word,
        stemmed_word=stemmed_word,
        alternatives=alternatives,
    )
