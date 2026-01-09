.PHONY: stage1 stage2 docs repo_map

stage1:
	python -m pip install -e stage1_dataset
	MPLBACKEND=Agg fly-olf-stage1 build stage1_dataset/configs/default.yaml
	$(MAKE) docs

stage2:
	python -m pip install -e stage2_baseline
	fly-olf-stage2 stage2_baseline/configs/default.yaml
	$(MAKE) docs

repo_map:
	python scripts/update_repo_map.py

docs: repo_map
	@echo "Docs updated."
