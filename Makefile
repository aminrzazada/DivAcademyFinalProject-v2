# Pipeline axını: setup → test → ingest → train → evaluate → app
# İstifadə: make all DATA=data/Base.csv     (Windows-da: Git Bash/WSL və ya əmrləri əl ilə işlədin)
PY   ?= python
DATA ?= data/Base.csv

.PHONY: help setup test synthetic ingest train train-sep ablation evaluate app all

help:
	@echo "make setup      - asılılıqları quraşdır"
	@echo "make test       - pytest (sintetik data, real data tələb etmir)"
	@echo "make ingest     - DATA=... faylını yoxla, results/data_quality.json və data/processed/base.parquet yaz"
	@echo "make train      - modeli təlim et (models/model.joblib, models/meta.json)"
	@echo "make train-sep  - calibration (ay 4) və threshold (ay 5) ayrı aylarda (models_sep/ qovluğuna)"
	@echo "make ablation   - velocity ablation (results/velocity_ablation.csv)"
	@echo "make evaluate   - 03_extended_analysis.ipynb-i işlət (real data lazımdır)"
	@echo "make app        - Streamlit app"
	@echo "make all        - test + ingest + train"

setup:
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -q tests

synthetic:
	$(PY) scripts/make_synthetic_baf.py 200000 data/Base_synthetic.csv

ingest:
	$(PY) -m src.ingest $(DATA) --parquet

train:
	$(PY) -m src.train $(DATA) models

train-sep:
	$(PY) -m src.train $(DATA) models_sep --separate-valid

ablation:
	$(PY) scripts/velocity_ablation.py $(DATA)

evaluate:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace 03_extended_analysis.ipynb

app:
	streamlit run app.py

all: test ingest train
