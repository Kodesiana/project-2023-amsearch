format:
	yapf -i -r -vv --style=pep8 .

dev:
	flask --app amsearch run --reload --debug

.PHONY: format dev
