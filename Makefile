IMAGE_NAME = denoising_score_matching

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --rm -v $(shell pwd)/output:/app/output $(IMAGE_NAME)

up: build run

clean:
	docker rmi $(IMAGE_NAME)

.PHONY: build run up clean
