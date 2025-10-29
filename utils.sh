
python train_syllable_rhythm_model.py LJSpeech /home/berta/data/LJSpeech/wav_preprocess_pydub/ /home/berta/workspace/RnV/LJSpeech/LJSpeech-segmenter.pth LJSpeech

nohup bash -c 'for i in C_{001..050}; do python train_urhythmic_rhythm_model_by_speaker.py "$i" global HunDys/HunDys-wavlm/ Szindbad/Szindbad-segmenter.pth HunDys/; done' > nohup_train_uryglobal_HunDys.log 2>&1 &
nohup bash -c 'for i in C_{001..050}; do python train_syllable_rhythm_model_by_speaker.py "$i" /home/berta/data/HungarianDysartriaDatabase/preprocessed_rnv_wav/ Szindbad/Szindbad-segmenter.pth HunDys/; done' > nohup_train_syllable_HunDys.log 2>&1 &