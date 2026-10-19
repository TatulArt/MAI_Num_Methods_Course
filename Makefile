PYTHON = .venv/bin/python

.PHONY: setup lab1 1.1 1.2 1.3 1.4 1.5

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

1.1:
	cd lab1 && ../$(PYTHON) task1_1_lu.py

1.2:
	cd lab1 && ../$(PYTHON) task1_2_thomas.py

1.3:
	cd lab1 && ../$(PYTHON) task1_3_iterations.py

1.4:
	cd lab1 && ../$(PYTHON) task1_4_jacobi.py

1.5:
	cd lab1 && ../$(PYTHON) task1_5_qr.py

lab1: 1.1 1.2 1.3 1.4 1.5
