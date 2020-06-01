test:
	python -mpytest src/ src/tests/

cov:
	python -mpytest --cov-config=setup.cfg --cov-report html --cov=src src/ src/tests/
