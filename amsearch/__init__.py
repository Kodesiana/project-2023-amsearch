from dotenv import load_dotenv
load_dotenv("amsearch/.env")

from flask import Flask

from amsearch.db import db, User
from amsearch.login import lm
from amsearch.embeddings import Embeddings, Stemmer
from amsearch.services import SearchService

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


# load embedding models
embedding_service = Embeddings()
embedding_service.load(app.config['DATA_DIR'])

# create stemmer
stemming_service = Stemmer()
stemming_service.load(app.config['DATA_DIR'])

# create search engine
search_service = SearchService(app.config['GOOGLE_CSE_ID'],
                            app.config['GOOGLE_CSE_API_KEY'],
                            embedding_service)

# register blueprints
from amsearch.controllers import auth, admin, search

app.register_blueprint(auth.router)
app.register_blueprint(admin.router)
app.register_blueprint(search.router)
