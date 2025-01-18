IMAGE_VERSION=$(file < VERSION)

format:
	yapf -i -r -vv --style=pep8 .

dev:
	flask --app amsearch run --reload --debug

build:
	docker build -t fahminlb33/project-2023-amsearch:v${IMAGE_VERSION} amsearch
	docker push fahminlb33/project-2023-amsearch:v${IMAGE_VERSION}

.PHONY: format dev
