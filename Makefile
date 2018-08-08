.PHONY: install remove dev voice_preprocess

install:
	python setup.py bdist_wheel
	sudo pip2 install ./dist/*.whl

remove:
	rm -rf build *.egg-info dist
	sudo pip2 uninstall ciessl_py_pkgs

dev:
	sudo pip2 install -e .

voice_preprocess:
	python ciessl_app/voice_preprocess.py --data_in="data/sample" --data_out="data/active_voice"