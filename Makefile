.PHONY: install remove dev voice_preprocess train save_gccphat \
	train_autoencoder test_enc wav2pickle save_stft

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

save_stft:
	python ciessl_app/tools/save_stft.py --voice="data/hand_vad_pickle/test" --map="data/map/bh9f_lab_map.json" \
		--config="ciessl_app/config/bh9f_pos_tf.json" --out="data/stft_data/test"

save_gccphat:
	python ciessl_app/tools/save_gccphat.py --voice="data/active_voice" --map="data/map/bh9f_lab_map.json" \
		--config="ciessl_app/config/bh9f_pos_tf.json" --out="data/gccphat"

save_conv_ae_code:
	python ciessl_app/tools/save_conv_ae_code.py --data="data/stft_data/train/amp" --out="data/conv_code_256_16" \
		--model="data/model/stft_cae_subset_256.json"

train_autoencoder:
	python ciessl_app/train_autoencoder.py --data="data/conv_code_256_16" --encoder="denoise_ae" \
		--out="denoise_ae_inner.model"

train:
	python ciessl_app/train.py --voice_data="data/hand_vad_pickle/test/3Room" --map_data="data/map/bh9f_lab_map.json" \
		--config="ciessl_app/config/bh9f_3room.json" --mode="rank" --voice_feature="conv_enc" --model_type="haram" \
		--voice_encoder="./data/model/stft_cae_subset_256.json" --map_feature="flooding" --n_mic=16 \
		--save_train_hist="data/results/HARAM+AE" --n_trails=1 --save_trace="data/results/HARAM+AE/trace"

visualize:
	python ciessl_app/visualizer.py --data "data/results/HARAM+AE_16mic/acc" "data/results/haram-gcc/acc" --out="acc_errorband" --plot="acc_variance" \
		--name_tag "HARAM+AE (Ours)" "HARAM+GCC"

test_enc:
	python ciessl_app/tools/test_enc.py --dataset="data/stft_data/train/amp" \
		--encoder_model="data/model/stft_cae_subset.json"

