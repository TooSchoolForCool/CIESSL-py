.PHONY: install remove dev voice_preprocess train save_gccphat \
	train_autoencoder test_enc wav2pickle

install:
	python setup.py bdist_wheel
	pip install ./dist/*.whl

remove:
	rm -rf build *.egg-info dist
	pip uninstall ciessl_py_pkgs

dev:
	pip install -e .


wav2pickle:
	python ciessl_app/tools/wav2pickle.py --data_in="data/hand_vad_wav" --data_out="data/hand_vad_pickle" \

voice_preprocess:
	python ciessl_app/tools/voice_preprocess.py --data_in="data/raw_voice" --data_out="data/active_voice" \
		--chunk_interval=20 --mode=3

save_gccphat:
	python ciessl_app/tools/save_gccphat.py --voice="data/active_voice" --map="data/map/bh9f_lab_map.json" \
		--config="ciessl_app/config/bh9f_pos_tf.json" --out="data/gccphat"

train_autoencoder:
	python ciessl_app/train_autoencoder.py --data="data/active_voice" --encoder="voice_ae" \
		--out="raw_voice_ae.model"

train:
	python ciessl_app/train.py --voice_data="data/active_voice" --map_data="data/map/bh9f_lab_map.json" \
		--config="ciessl_app/config/bh9f_pos_tf.json" --mode="clf" --voice_feature="enc" \
		--voice_encoder="./data/model/raw_voice_ae_1.json" --map_feature="flooding"
test_enc:
	python ciessl_app/test_enc.py --dataset="data/gccphat" --encoder_model="data/model/gccphat_ae_1.json"