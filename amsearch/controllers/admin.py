import uuid

from flask import Blueprint, flash, render_template, redirect, url_for, request
from flask_login import login_required

from amsearch.services import VectorSearchInstance
from amsearch.db import db, Document

router = Blueprint("admin", __name__)


@router.route("/admin")
@login_required
def list():
    q = request.args.get("q", "")
    if not q:
        pagination = db.paginate(
            db.select(Document).order_by(Document.published_at.desc()), per_page=10
        )
    else:
        pagination = db.paginate(
            db.select(Document)
            .filter(Document.title.ilike(f"%{q}%"))
            .order_by(Document.published_at.desc()),
            per_page=10,
        )

    return render_template(
        "pages/admin/list.html", pagination=pagination, search_term=q
    )


@router.route("/admin/create")
@login_required
def create():
    return render_template("pages/admin/edit.html")


@router.route("/admin/edit/<id>")
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
    

@router.route("/admin/remove/<id>")
@login_required
def remove(id: str):
    # find and delete from db
    doc = db.get_or_404(Document, id)
    title = doc.title

    db.session.delete(doc)
    db.session.commit()

    # flash message
    flash(f"Data berhasil dihapus!<br><strong>{ title }</strong>", "success")

    return redirect(url_for("admin.list"))


@router.route("/admin/save", methods=["POST"])
@login_required
def save():
    try:
        # get the fields
        title = request.form.get("title")
        content = request.form.get("content")
        url = request.form.get("url")
        published_at = request.form.get("published_at")

        # stem the content
        content_stemmed, count = VectorSearchInstance.stem_sentence(content, "ams")

        # make embeddings
        eb = VectorSearchInstance.embed(content_stemmed)

        # check if this is an edit operation
        id = request.form.get("id")
        if id:
            doc = db.get_or_404(Document, id)
            doc.title = title
            doc.content = content
            doc.token_count = count
            doc.source_url = url
            doc.published_at = published_at
            doc.embedding_bert = eb
            doc.embedding_tfidf = eb
        else:
            # create row
            doc = Document(
                id=str(uuid.uuid4()),
                title=title,
                content=content_stemmed,
                token_count=count,
                source_url=url,
                published_at=published_at,
                embedding_bert=eb,
                embedding_tfidf=eb,
            )

        # add to db
        if not id:
            db.session.add(doc)

        # save to db
        db.session.commit()

        # flash message
        flash(f"Data berhasil ditambahkan!<br><strong>{ title }</strong>", "success")
        return redirect(url_for("admin.list"))
    except:
        flash(f"Data gagal disimpan!<br>Pastikan semua kolom sudah diisi.", "danger")
        return render_template("pages/admin/edit.html",
            id=id,
            title=title,
            content=content,
            url=url,
            published_at=published_at,
        )

