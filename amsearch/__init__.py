from flask import Flask, render_template

from amsearch.db import db, User
from amsearch.login import lm
from amsearch.services import VectorSearchInstance

# create and configure the app
app = Flask(__name__)
app.config.from_prefixed_env()

# create the database
db.init_app(app)

# create session manager
lm.init_app(app)


# Flask-Login user loader
@lm.user_loader
def load_user(user_id):
    return User.query.get(user_id)


# load stemmers and embedding models
VectorSearchInstance.load(app.config["DATA_DIR"])

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
    return render_template("pages/public/about.html")
