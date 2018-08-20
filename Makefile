.PHONY: install remove dev voice_preprocess train

install:
	python setup.py bdist_wheel
	pip install ./dist/*.whl

remove:
	rm -rf build *.egg-info dist
	pip uninstall ciessl_py_pkgs

dev:
	pip install -e .

voice_preprocess:
	python ciessl_app/tools/voice_preprocess.py --data_in="data/raw_voice" --data_out="data/active_voice" \
		--chunk_interval=20 --mode=3

train_autoencoder:
	python ciessl_app/train_autoencoder.py --voice="data/active_voice" --map="data/map/bh9f_lab_map.json" \
		--config="ciessl_app/config/bh9f_pos_tf.json" --encoder="all_ch_vae" --out="all_ch_vae.model"

train:
	python ciessl_app/train.py --voice_data="data/active_voice" --map_data="data/map/bh9f_lab_map.json" \
		--config="ciessl_app/config/bh9f_pos_tf.json" --mode="clf" --voice_feature="enc" \
		--voice_encoder="data/model/all_ch_vae.model"

cmp_enc:
	python ciessl_app/cmp_encode.py --voice_data="data/active_voice" --map_data="data/map/bh9f_lab_map.json" \
		--config="ciessl_app/config/bh9f_pos_tf.json" --voice_encoder="data/model/all_ch_vae.model"