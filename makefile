.ONESHELL:
.DEFAULT_GOAL := run

# -----------------------------------------------------------------------------
# pybind11 DStarLite build
# -----------------------------------------------------------------------------
PYTHON_INCLUDES := $(shell python3.12-config --includes) $(shell python3 -m pybind11 --includes)
EXT_SUFFIX      := $(shell python3-config --extension-suffix)
DSTAR_SRC       := src/DStarLite/DStarLite.cpp
DSTAR_HDR       := src/DStarLite/DStarLite.h
DSTAR_OUT       := src/DStarLite/DStarLite$(EXT_SUFFIX)

CXXFLAGS       += -O3 -std=c++17 -fPIC -Isrc/DStarLite $(PYTHON_INCLUDES)
LDFLAGS        += -shared

dstar: $(DSTAR_OUT)

$(DSTAR_OUT): $(DSTAR_SRC) $(DSTAR_HDR) src/DStarLite/__init__.py
	@echo "Building C++ D* Lite extension $@"
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@

run r: dstar
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

copy_all cpa:
	scp -r kysp2d@mill.mst.edu:coles_custom_Grid/custom_Grid/saved_experiments /home/student/REU/custom_Grid/

tensorboard tb:
	pkill tensorboard || true
	source venv/bin/activate && \
	tensorboard --logdir saved_experiments --port 6006 & \
	sleep 4 && \
	xdg-open http://localhost:6006/

tensorboard tb_old:
	pkill tensorboard || true
	source venv/bin/activate && \
	tensorboard --logdir old_experiments --port 6006 & \
	sleep 4 && \
	xdg-open http://localhost:6006/

archive_experiments ae:
	@echo "Archiving experiments to old_experiments/..."
	TIMESTAMP=$$(date +%Y%m%d_%H%M%S) ; \
	mkdir -p old_experiments/$$TIMESTAMP ; \
	mv saved_experiments/* old_experiments/$$TIMESTAMP/ ; \
	echo "Moved to old_experiments/$$TIMESTAMP"

clean_experiments ce:
	@echo "Cleaning contents of saved_experiments/..."
	rm -rf saved_experiments/*
	rm -rf saved_experiments/.*
	@echo "Finished cleaning saved_experiments"

mill m:
	srun -p gpu --gres gpu:V100-SXM2-32GB:1 -n 8 -N 1 --mem=64G --time=48:00:00 --pty bash


