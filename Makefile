all: build

build:
	bash debugging.sh
	echo "PING!"
	cat ./archive_name
run:
	bash run.sh
train:
	bash train.sh
