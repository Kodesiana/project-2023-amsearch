from flask import Blueprint, render_template, redirect, flash, url_for, request
from flask_login import login_required, login_user, logout_user
from werkzeug.security import check_password_hash

from amsearch.db import db, User

router = Blueprint("auth", __name__)


@router.route("/login", methods=["GET", "POST"])
def login():
    # render login page
    if request.method == "GET":
        return render_template("pages/auth/login.html")

    # get user data
    username = request.form.get("username")
    password = request.form.get("password")

    # find the user by email
    query = db.select(User).where(User.username == username)
    user = db.session.execute(query).scalar_one_or_none()

    # verify password
    if not user or not check_password_hash(user.hashed_password, password):
        flash("Email atau password salah")
        return redirect(url_for("auth.login"))

    # login user
    login_user(user, remember=True)

    # redirect to home page
    return redirect(url_for("admin.list"))


@router.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("search.home"))
