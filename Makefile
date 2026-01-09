.PHONY: stage1

stage1:
	python -m pip install -e stage1_dataset
	MPLBACKEND=Agg fly-olf-stage1 stage1_dataset/configs/default.yaml
