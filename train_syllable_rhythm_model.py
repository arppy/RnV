# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: GPL-3.0-only

import argparse
import os
from pathlib import Path

import librosa
import torch
from scipy import stats
from tqdm import tqdm

from rnv.rhythm.syllable.syllable_segmenter import SyllableSegmenter
from rnv.ssl.models import WavLM


def get_speaker_peak_to_peak_and_silence_durations(syllable_segmenter, audio_filepaths, feats_dir):
    speaker_peak_to_peak_durations_in_s = []
    speaker_silence_durations_in_s = []

    for audio_path in tqdm(audio_filepaths):
        wav, sr = librosa.load(audio_path, sr=16000)
        feat_path = feats_dir / audio_path.with_suffix(".pt").name
        if not feat_path.exists():
            print(f"Warning: No feature file found at {feat_path}. Skipping.")
            continue
        feats = torch.load(feat_path, weights_only=True).cpu()
        peak_to_peak_durations_in_s, silence_durations_in_s = syllable_segmenter.get_audio_peak_to_peak_and_silence_durations(wav, feats)

        speaker_peak_to_peak_durations_in_s.extend(peak_to_peak_durations_in_s)
        speaker_silence_durations_in_s.extend(silence_durations_in_s)

    return speaker_peak_to_peak_durations_in_s, speaker_silence_durations_in_s


def compute_speaker_rhythm_model(speaker_id, audio_data_path, feats_dir, segmenter_checkpoint, output_dir):
    syllable_segmenter = SyllableSegmenter(urhythmic_segmenter_checkpoint_path=segmenter_checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    audio_filepaths = list(Path(audio_data_path).rglob("*.wav"))
    if len(audio_filepaths) == 0:
        print("No audio files found for speaker.")
        return

    speaker_peak_to_peak_durations_in_s, speaker_silence_durations_in_s = get_speaker_peak_to_peak_and_silence_durations(syllable_segmenter, audio_filepaths, feats_dir)
    # --- ADD SAFETY CHECKS HERE ---
    if len(speaker_peak_to_peak_durations_in_s) == 0:
        print(f"Error: No syllables detected in any audio file for speaker {speaker_id}. Skipping model saving.")
        return

    total_syllable_duration = sum(speaker_peak_to_peak_durations_in_s)
    if total_syllable_duration <= 0:
        print(f"Error: Total syllable duration is zero or negative for speaker {speaker_id}. Skipping.")
        return

    speaker_speaking_rate = len(speaker_peak_to_peak_durations_in_s) / total_syllable_duration
    print("Speaking rate:", speaker_speaking_rate)

    syllable_shape, floc, syllable_scale = stats.gamma.fit(speaker_peak_to_peak_durations_in_s, floc=0)
    print("Syllable Gamma distribution:", syllable_shape, syllable_scale)

    silence_shape, floc, silence_scale = stats.gamma.fit(speaker_silence_durations_in_s, floc=0)
    print("Silence Gamma distribution:", silence_shape, silence_scale)

    checkpoint_path = f"{output_dir}/{speaker_id}_syllable_models.pth"
    torch.save(
        {"speaking_rate": speaker_speaking_rate, "syllable_shape": syllable_shape, "syllable_scale": syllable_scale, "silence_shape": silence_shape, "silence_scale": silence_scale},
        checkpoint_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Syllable rhythm model for speaker.")
    parser.add_argument(
        "speaker_id",
        metavar="speaker-id",
        help="Speaker ID.",
        type=Path,
    )
    parser.add_argument(
        "audio_dir",
        metavar="audio-dir",
        help="path to the speaker's audio data directory.",
        type=Path,
    )
    parser.add_argument("feats_dir", metavar="feats-dir", type=Path)
    parser.add_argument(
        "segmenter_checkpoint_path",
        metavar="segmenter-checkpoint-path",
        help="path to the segmenter model checkpoint.",
        type=Path,
    )
    parser.add_argument("out_dir", metavar="out-dir", type=Path, help="path to the output directory.")
    args = parser.parse_args()

    compute_speaker_rhythm_model(args.speaker_id, args.audio_dir, args.feats_dir, args.segmenter_checkpoint_path, args.out_dir)
