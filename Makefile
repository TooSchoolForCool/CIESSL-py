.PHONY: install remove dev voice_preprocess train

install:
	python setup.py bdist_wheel
	sudo pip2 install ./dist/*.whl

remove:
	rm -rf build *.egg-info dist
	sudo pip2 uninstall ciessl_py_pkgs

dev:
	sudo pip2 install -e .

voice_preprocess:
	python ciessl_app/voice_preprocess.py --data_in="data/raw_voice" --data_out="data/active_voice" \
		--chunk_interval=20 --mode=3

train:
	python ciessl_app/train.py --voice="data/active_voice" --map="data/map"