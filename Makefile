.PHONY: install remove dev

install:
	python setup.py bdist_wheel
	sudo pip2 install ./dist/*.whl

remove:
	rm -rf build *.egg-info dist
	sudo pip2 uninstall ciessl_py_pkgs

dev:
	sudo pip2 install -e .