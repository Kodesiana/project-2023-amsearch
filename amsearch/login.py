from flask_login import LoginManager

lm = LoginManager()
lm.login_view = "auth.login"
