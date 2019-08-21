.PHONY: build run

image=g2t

build:
	docker build -t $(image) .

run: 
	docker run -p 6005:6006 -it --rm=true -v $(PWD):/g2t $(image) bash -l


run_gpu:
	docker run -p 6005:6006 --runtime=nvidia -it --rm=true -v $(PWD):/g2t $(image) bash -l 


run_gpu0:
	docker run --runtime=nvidia -it --rm=true -e NVIDIA_VISIBLE_DEVICES=0 -v $(PWD):/g2t $(image) bash -l 

run_gpu1:
	docker run --runtime=nvidia -it --rm=true -e NVIDIA_VISIBLE_DEVICES=1 -v $(PWD):/g2t $(image) bash -l 

undocker:
	docker stop $(image)
	docker rm $(image)
