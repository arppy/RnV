
python train_syllable_rhythm_model.py LJSpeech /home/berta/data/LJSpeech/wav_preprocess_pydub/ /home/berta/workspace/RnV/LJSpeech/LJSpeech-segmenter.pth LJSpeech

nohup bash -c 'for i in C_{001..050}; do python train_urhythmic_rhythm_model_by_speaker.py "$i" global HunDys/HunDys-wavlm/ Szindbad/Szindbad-segmenter.pth HunDys/; done' > nohup_train_uryglobal_HunDys.log 2>&1 &
nohup bash -c 'for i in C_{001..050}; do python train_syllable_rhythm_model_by_speaker.py "$i" /home/berta/data/HungarianDysartriaDatabase/preprocessed_rnv_wav/ Szindbad/Szindbad-segmenter.pth HunDys/; done' > nohup_train_syllable_HunDys.log 2>&1 &


nohup bash -c 'for i in C_{001..050}; do python train_urhythmic_rhythm_model_by_speaker.py "$i" global HunDysSepformer/HunDys-wavlm/ Szindbad/Szindbad-segmenter.pth HunDysSepformer/; done' > nohup_train_uryglobal_HunDysSepformer.log 2>&1 &
nohup bash -c 'for i in C_{001..050}; do python train_syllable_rhythm_model_by_speaker.py "$i" /home/berta/workspace/DysartriaClassifier/HunDys/prepHunDysSepFormerDNS4_wav/ Szindbad/Szindbad-segmenter.pth HunDysSepformer/; done' > nohup_train_syllable_HunDysSepformer.log 2>&1 &



nohup bash -c  'for i in C_{001..050}; do CUDA_VISIBLE_DEVICES=3 python convert_by_speaker.py "$i" Szindbad syllable fine HunDys/HunDys-wavlm/ Szindbad/Szindbad-wavlm/ /home/berta/models/vocoder-hifigan/wavlm-hifigan-prematch_g_02500000/ Szindbad/Szindbad-segmenter.pth knnvc HunDys/HunDys_knnvc_syllable_fine_wav /home/berta/data/HungarianDysartriaDatabase/preprocessed_rnv_wav/; done' > nohup_convert_syllablefine_HunDys.log 2>&1 &