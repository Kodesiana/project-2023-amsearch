import os
import json

from flask import request, session


class Localization:
    def __init__(self, app=None):
        if app is not None:
            self.app = app
            self.init_app(app)

        self._messages = {}

    def init_app(self, app):
        self.app = app
        app.config.setdefault("MESSAGES_DIR", "i18n")
        app.config.setdefault("DEFAULT_LANGUAGE", "id")

        app.context_processor(lambda: dict(site_locale=self.get_locale()))
        app.jinja_env.globals.update(get_message=self.get_message, _=self.get_message)

        # load translations
        for language in self.get_locales():
            filepath = os.path.join(
                self.app.config.get("MESSAGES_DIR"), f"{language}.json"
            )
            with open(filepath, "r", encoding="utf-8") as f:
                self._messages[language] = json.load(f)

    def get_locales(self):
        return [
            x.replace(".json", "")
            for x in os.listdir(self.app.config.get("MESSAGES_DIR"))
        ]

    def get_locale(self):
        return (
            session.get("language")
            or request.accept_languages.best_match(self.get_locales())
            or self.app.config.get("DEFAULT_LANGUAGE")
        )

    def set_locale(self, language=None):
        session["language"] = language or self.app.config.get("DEFAULT_LANGUAGE")

    def get_message(self, message_code, *args):
        language = self.get_locale()

        if language in self._messages:
            message = self._messages[language].get(message_code, f"<{message_code}>")
        else:
            message = self._messages[self.app.config.get("DEFAULT_LANGUAGE")].get(
                message_code, f"<{message_code}>"
            )

        return message.format(*args)


i18n = Localization()
