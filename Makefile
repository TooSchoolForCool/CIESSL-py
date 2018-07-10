.PHONY: install remove

install:
	python setup.py bdist_wheel
	sudo pip2 install ./dist/*.whl

remove:
	rm -rf build ciessl.egg-info dist
	sudo pip2 uninstall ciessl

dev:
	sudo pip2 install -e .