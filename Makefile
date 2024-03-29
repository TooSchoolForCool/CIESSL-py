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


save_gccfb:
	python ciessl_app/tools/save_gccfb.py --voice="data/hand_vad_pickle/test/3Room" --map="data/map/bh9f_lab_map.json" \
		--config="ciessl_app/config/bh9f_3room.json" --out="data/gccfb"


save_conv_ae_code:
	python ciessl_app/tools/save_conv_ae_code.py --data="data/stft_data/train/amp" --out="data/conv_code_256_16" \
		--model="data/model/stft_cae_subset_256.json"


parse_trace:
	python ciessl_app/tools/parse_trace.py --data="data/trace/3room_trace.json" --out="data/trace_hist"


train_autoencoder:
	python ciessl_app/train_autoencoder.py --data="data/conv_code_256_16" --encoder="denoise_ae" \
		--out="denoise_ae_inner.model"

train_gccfb:
	python ciessl_app/train_gccfb.py --data="data/gccfb" --map="data/map/bh9f_lab_map.json"
	
train:
	python ciessl_app/train.py --voice_data="data/hand_vad_pickle/test/3Room" --map_data="data/map/bh9f_lab_map.json" \
		--config="ciessl_app/config/bh9f_3room.json" --mode="rl" --voice_feature="conv_enc" --model_type="haram" \
		--voice_encoder="./data/model/stft_cae_subset_256.json" --map_feature="flooding" --n_mic=16 --lm_param=0.965 \
		--save_train_hist="HARAM+AE" --save_trace="HARAM+AE/trace" --n_trails=1 

visualize:
	python ciessl_app/visualizer.py --data "HARAM-GCCFB/acc" --out="acc_errorband" --plot="acc_variance"

test_enc:
	python ciessl_app/tools/test_enc.py --dataset="data/stft_data/train/amp" \
		--encoder_model="data/model/stft_cae_subset.json"

