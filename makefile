# Makefile

VENV_NAME := venv
PYTHON_INTERPRETER := python3

.PHONY: venv install run

venv:
	@echo "Creating virtual environment..."
	$(PYTHON_INTERPRETER) -m venv $(VENV_NAME)

install: venv
	@echo "Installing dependencies..."
	$(VENV_NAME)/bin/pip install -r requirements.txt

run: install
	@echo "Running main.py..."
	$(VENV_NAME)/bin/python scr/main.py

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_NAME)

.PHONY: clean
