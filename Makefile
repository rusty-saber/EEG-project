.PHONY: setup test download preprocess train-stage1 train-stage2 evaluate mvp clean

# Setup
setup:
	pip install -r requirements.txt

# Testing
test:
	pytest tests/ -v

# Data Pipeline
download:
	python scripts/download_data.py --dataset physionet --subjects 5

download-full:
	python scripts/download_data.py --dataset physionet --subjects 109

preprocess:
	python scripts/preprocess.py --dataset physionet

# Training
train-stage1:
	python scripts/train.py --config configs/stage1.yaml

train-stage2:
	python scripts/train.py --config configs/stage2.yaml --resume outputs/checkpoints/stage1_best.pt

# Evaluation
evaluate:
	python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt

# MVP - Single command to run full minimal experiment
mvp:
	python scripts/run_mvp.py

# Cleanup
clean:
	rm -rf outputs/checkpoints/* outputs/logs/* outputs/figures/*

clean-data:
	rm -rf data/processed/*
