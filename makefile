.ONESHELL:

run r:
	source venv/bin/activate && python main.py

git g:
	git add .
	git commit -m 'sync'
	git push

reset_venv rv:
	@echo "Resetting venv..."
	rm -rf venv
	python3 -m venv venv
	source venv/bin/activate
	python -m ensurepip
	pip install --upgrade pip
	mkdir -p $$HOME/.cache/pip-tmp
	TMPDIR=$$HOME/.cache/pip-tmp pip install -r requirements.txt
	@echo "Finished resetting logs"

copy_logs cp:
	scp -r cs3f7@mill.mst.edu:~/custom_Grid/logs /home/student/REU/custom_Grid/

clean_logs cl:
	@echo "Cleaning contents of logs/..."
	rm -rf logs/*
	rm -rf logs/.*
	@echo "Finished cleaning logs"

tensorboard tb:
	pkill tensorboard || true
	source venv/bin/activate && \
	tensorboard --logdir logs --port 6006 & \
	sleep 2 && \
	xdg-open http://localhost:6006/

tb_saved:
	pkill tensorboard || true
	source venv/bin/activate && \
	tensorboard --logdir saved_logs --port 6006 & \
	sleep 2 && \
	xdg-open http://localhost:6006/
