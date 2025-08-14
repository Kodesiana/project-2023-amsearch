from flask import Flask, render_template, redirect, url_for, request

from amsearch.db import db, User
from amsearch.login import lm
from amsearch.services import IR
from amsearch.localization import i18n

# create and configure the app
app = Flask(__name__)
app.config.from_prefixed_env()

# create the database
db.init_app(app)

# create session manager
lm.init_app(app)

# initialize localization
i18n.init_app(app)


# Flask-Login user loader
@lm.user_loader
def load_user(user_id):
    return User.query.get(user_id)


# load stemmers and embedding models
IR.load(app.config["VOCAB_PATH"], app.config["MODEL_PATH"])

# register blueprints
from amsearch.controllers import auth, admin, search, stemming, statistics

app.register_blueprint(auth.router)
app.register_blueprint(admin.router)
app.register_blueprint(search.router)
app.register_blueprint(stemming.router)
app.register_blueprint(statistics.router)


@app.get("/")
def home():
    return render_template("pages/public/home.html")


@app.get("/about")
def about():
    if i18n.get_locale() == "id":
        return render_template("pages/public/about_id.html")
    
    return render_template("pages/public/about_en.html")
    


@app.get("/toggle-language")
def toggle_language():
    if i18n.get_locale() == "id":
        i18n.set_locale("en")
    else:
        i18n.set_locale("id")

    prev_url = request.args.get("to", url_for("home"))
    return redirect(prev_url)
