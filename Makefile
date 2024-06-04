.ONESHELL:

.PHONY: check format

SHELL = /bin/zsh
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
VENV = llava
PYTHON_SRC = data_visualization

format:
	$(CONDA_ACTIVATE) $(VENV)
	python -m black $(PYTHON_SRC)
	python -m isort $(PYTHON_SRC)

check:
	$(CONDA_ACTIVATE) $(VENV)
	python -m black --check $(PYTHON_SRC)
	python -m isort --check-only $(PYTHON_SRC)
	python -m mypy $(PYTHON_SRC)
	python -m pylint $(PYTHON_SRC) -f parseable -r n

