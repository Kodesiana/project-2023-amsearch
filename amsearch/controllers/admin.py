import uuid

from flask import Blueprint, render_template, redirect, url_for, request
from flask_login import login_required

from amsearch import embedding_service, stemming_service
from amsearch.db import db, Document

router = Blueprint('admin', __name__)


@router.route('/admin')
@login_required
def list():
    pagination = db.paginate(db.select(Document).order_by(Document.title))
    return render_template("pages/admin/list.html", pagination=pagination)


@router.route('/admin/create')
@login_required
def create():
    return render_template("pages/admin/edit.html")


@router.route('/admin/edit/<id>')
@login_required
def update(id: str):
    doc = db.get_or_404(Document, id)
    return render_template("pages/admin/edit.html",
                           id=id,
                           title=doc.title,
                           content=doc.content,
                           url=doc.source_url,
                           published_at=doc.published_at)


@router.route('/admin/remove/<id>')
@login_required
def remove(id: str):
    # find and delete from db
    doc = db.get_or_404(Document, id)
    db.session.delete(doc)
    db.session.commit()

    return redirect(url_for('admin.list'))


@router.route('/admin/save', methods=['POST'])
@login_required
def save():
    # get the fields
    title = request.form.get('title')
    content = request.form.get('content')
    url = request.form.get('url')
    published_at = request.form.get('published_at')

    # stem the content
    content_stemmed, count = stemming_service.stem_sentence(content)

    # make embeddings
    eb = embedding_service.extract_bert(content_stemmed)
    et = embedding_service.extract_tfidf(content_stemmed)

    # check if this is an edit operation
    id = request.form.get('id')
    if id:
        doc = db.get_or_404(Document, id)
        doc.title = title
        doc.content = content
        doc.token_count = count
        doc.source_url = url
        doc.published_at = published_at
        doc.embedding_bert = eb
        doc.embedding_tfidf = et
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
            embedding_tfidf=et,
        )

    # add to db
    if not id:
        db.session.add(doc)

    # save to db
    db.session.commit()

    return redirect(url_for('admin.list'))
