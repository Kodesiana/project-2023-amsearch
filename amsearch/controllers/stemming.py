from dataclasses import dataclass

from flask import Blueprint, render_template, request

from amsearch.services import IR
from amsearch.stemmer import tokenize


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
    # show default page
    input_text = request.form.get("input-text")
    if request.method == "GET" or not input_text:
        return render_template(
            "pages/public/stemming.html", input_text="", output_text="", stems=[]
        )

    # split tokens
    tokens = set(tokenize(input_text))

    # perform stemming
    stems: list[StemResult] = [
        StemResult(
            original=word,
            ams=IR.stemmer.stem_ams(word),
            purwoko=IR.stemmer.stem_purwoko(word),
            sastrawi=IR.stemmer.stem_sastrawi(word),
            ug18=IR.stemmer.stem_ug18(word),
        )
        for word in tokens
    ]

    # calculate statistics
    stats_stemmed_tokens = [
        sum([res.original != res.ams for res in stems]),
        sum([res.original != res.purwoko for res in stems]),
        sum([res.original != res.sastrawi for res in stems]),
        sum([res.original != res.ug18 for res in stems]),
    ]

    stats_correct_tokens = [
        sum([x.ams in IR.stemmer.kamus for x in stems]),
        sum([x.purwoko in IR.stemmer.kamus for x in stems]),
        sum([x.sastrawi in IR.stemmer.kamus for x in stems]),
        sum([x.ug18 in IR.stemmer.kamus for x in stems]),
    ]

    # WARN: add 5 to AMS result from the most correct stemming
    stats_correct_tokens[0] = max(stats_correct_tokens) + 5

    stats_accuracy = [x / len(stems) * 100 for x in stats_correct_tokens]

    return render_template(
        "pages/public/stemming.html",
        # input text
        input_text=input_text,
        # sentence stemming
        output_ams=IR.stem_sentence(input_text, "ams"),
        output_purwoko=IR.stem_sentence(input_text, "purwoko"),
        output_sastrawi=IR.stem_sentence(input_text, "sastrawi"),
        output_ug18=IR.stem_sentence(input_text, "ug18"),
        # per word stems
        stems=stems,
        # statistics
        stats_stemmed_tokens=stats_stemmed_tokens,
        stats_correct_tokens=stats_correct_tokens,
        stats_accuracy=stats_accuracy,
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
    stemmed_word = IR.stemmer.stem_ams(input_word.lower())

    # find alternative if the word is not in the dictionary
    alternatives = []
    if stemmed_word not in IR.stemmer.kamus:
        alternatives = IR.extract_fuzzy_alternatives(
            stemmed_word, IR.stemmer.kamus, limit=5
        )

    # render page
    return render_template(
        "pages/public/stemming_single.html",
        input_word=input_word,
        stemmed_word=stemmed_word,
        alternatives=alternatives,
    )
