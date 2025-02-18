IMAGE_VERSION=$(file < VERSION)

format:
	ruff format amsearch scripts

dev:
	flask --app amsearch run --reload --debug

build:
	docker build -t fahminlb33/project-2023-amsearch:v${IMAGE_VERSION} amsearch

publish: build
	docker push fahminlb33/project-2023-amsearch:v${IMAGE_VERSION}

.PHONY: format dev build publish
