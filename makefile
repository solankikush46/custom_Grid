.ONESHELL:

run r:
	source venv/bin/activate && python main.py

git g:
	git add .
	git commit -m 'sync'
	git push

reset_venv rv:
	rm -rf venv
	python3 -m venv venv
	source venv/bin/activate
	python -m ensurepip
	pip install --upgrade pip
	mkdir -p $$HOME/.cache/pip-tmp
	TMPDIR=$$HOME/.cache/pip-tmp pip install -r requirements.txt

