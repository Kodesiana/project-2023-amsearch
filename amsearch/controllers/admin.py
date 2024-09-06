import csv
import uuid
from io import StringIO
from typing import Optional

from flask import Blueprint, flash, render_template, redirect, url_for, request
from flask_login import login_required
from sqlalchemy.exc import SQLAlchemyError

from amsearch.services import VectorSearchInstance
from amsearch.db import db, Document

router = Blueprint("admin", __name__)


def get_paginated_documents(search_term: Optional[str] = None, per_page: int = 10):
    query = db.select(Document).order_by(Document.published_at.desc())
    if search_term:
        query = query.filter(Document.title.ilike(f"%{search_term}%"))
    return db.paginate(query, per_page=per_page)


@router.route("/admin")
@login_required
def list():
    search_term = request.args.get("q", "")
    pagination = get_paginated_documents(search_term)
    return render_template(
        "pages/admin/list.html", pagination=pagination, search_term=search_term
    )


@router.route("/admin/create")
@login_required
def create():
    return render_template("pages/admin/edit.html")


@router.route("/admin/edit/<string:id>")
@login_required
def update(id: str):
    doc = db.get_or_404(Document, id)
    return render_template(
        "pages/admin/edit.html",
        id=id,
        title=doc.title,
        content=doc.content,
        url=doc.source_url,
        published_at=doc.published_at,
    )


@router.route("/admin/remove/<string:id>")
@login_required
def remove(id: str):
    try:
        doc = db.get_or_404(Document, id)
        db.session.delete(doc)
        db.session.commit()
        flash(f"Data berhasil dihapus!<br><strong>{ doc.title }</strong>", "success")
    except SQLAlchemyError as e:
        db.session.rollback()
        flash(f"Gagal menghapus dokumen: {str(e)}", "danger")
    return redirect(url_for("admin.list"))


@router.route("/admin/save", methods=["POST"])
@login_required
def save():
    form_data = request.form.to_dict()
    id = form_data.pop("id", None)

    try:
        content_stemmed, count = VectorSearchInstance.stem_sentence(
            form_data["content"], "ams"
        )
        embedding = VectorSearchInstance.embed(content_stemmed)

        if id:
            doc = db.get_or_404(Document, id)
            for key, value in form_data.items():
                setattr(doc, key, value)
        else:
            doc = Document(id=str(uuid.uuid4()), **form_data)

        doc.content = content_stemmed
        doc.token_count = count
        doc.embedding_bert = doc.embedding_tfidf = embedding

        if not id:
            db.session.add(doc)
        db.session.commit()

        flash(
            f"Data berhasil ditambahkan!<br><strong>{ doc.title }</strong>", "success"
        )
        return redirect(url_for("admin.list"))
    except SQLAlchemyError as e:
        db.session.rollback()
        flash(f"Data gagal disimpan!<br>Pastikan semua kolom sudah diisi.", "danger")
    except Exception as e:
        flash(f"Kesalahan tidak diketahui: {str(e)}", "danger")

    return render_template("pages/admin/edit.html", **form_data)


@router.route("/admin/download")
@login_required
def download():
    # create in-memory storage
    si = StringIO()

    # create csv writer
    cw = csv.writer(si)

    # get all documents
    rows = db.session.execute(
        db.select(
            Document.id,
            Document.title,
            Document.content,
            Document.source_url,
            Document.token_count,
            Document.published_at,
        )
    ).all()

    # write header
    cw.writerow(["id", "title", "content", "source_url", "token_count", "published_at"])

    # print each row
    for row in rows:
        cw.writerow(
            [
                row.id,
                row.title,
                row.content,
                row.source_url,
                row.token_count,
                row.published_at,
            ]
        )

    # return generator and header
    return si.getvalue(), {
        "Content-Type": "text/csv",
        "Content-Disposition": "attachment; filename=artikel-sunda.csv",
    }
