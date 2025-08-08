from pathlib import Path

import torch
from tqdm import tqdm

import argparse

from rnv.converter import Converter
from rnv.ssl.models import WavLM
from rnv.utils import get_vocoder_checkpoint_path

def convert(
    src_speaker_id: str,
    tgt_speaker_id: str,
    model_class_str: str,
    model_type_str: str,
    src_feats_dir: Path,
    tgt_feats_dir: Path,
    vocoder_checkpoint_path: Path,
    segmenter_path: Path,
    knnvc: str,
    output_dir: Path,
):
    vocoder_checkpoint_path = get_vocoder_checkpoint_path(vocoder_checkpoint_path)

    # Initialize the converter with the vocoder checkpoint and rhythm conversion settings
    # You can choose between "urhythmic" or "syllable" for rhythm_converter
    # and "global" or "fine" for rhythm_model_type

    converter = Converter(vocoder_checkpoint_path, rhythm_converter=model_class_str, rhythm_model_type=model_type_str) # or "fine" for fine-grained rhythm conversion

    source_rhythm_model = f"{src_speaker_id}/{src_speaker_id}_{model_type_str}_{model_class_str}_model.pth"
    target_rhythm_model = f"{tgt_speaker_id}/{tgt_speaker_id}_{model_type_str}_{model_class_str}_model.pth"

    for feat_path in tqdm(list(src_feats_dir.rglob("*.pt"))):
        source_feats = torch.load(feat_path, map_location="cpu")
        # Rhythm and Voice Conversion
        knnvc_topk = 4
        lambda_rate = 1.
        save_path = output_dir / feat_path.name
        save_path = save_path.with_suffix(".wav")
        if knnvc == "knnvc" :
            converter.convert(source_feats, tgt_feats_dir, source_rhythm_model, target_rhythm_model, segmenter_path, knnvc_topk, lambda_rate, save_path=save_path)
        elif knnvc == "knnvc-only" :
            # KnnVc Voice Conversion Only (Without Rhythm Conversion)
            converter.convert(source_feats, tgt_feats_dir, None, None, segmenter_path, knnvc_topk, lambda_rate, save_path=save_path)
        else :
            # Rhythm Conversion Only
            converter.convert(source_feats, None, source_rhythm_model, target_rhythm_model, segmenter_path, save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert.")
    parser.add_argument(
        "src_speaker_id",
        metavar="src-speaker-id",
        help="Source Speaker ID.",
        type=str,
    )
    parser.add_argument(
        "tgt_speaker_id",
        metavar="tgt-speaker-id",
        help="Target Speaker ID.",
        type=str,
    )
    parser.add_argument(
        "model_class",
        metavar="model-class",
        help="rhythm model class (urhythmic, syllable).",
        type=str,
        choices=["urhythmic", "syllable"],
    )
    parser.add_argument(
        "model_type",
        metavar="model-type",
        help="rhythm model type (global, fine).",
        type=str,
        choices=["global", "fine"],
    )
    parser.add_argument(
        "src_feats_dir",
        metavar="src-feats-dir",
        help="path to the directory of source speaker's feature files (*.pt).",
        type=Path,
    )
    parser.add_argument(
        "tgt_feats_dir",
        metavar="tgt-feats-dir",
        help="path to the directory of target speaker's feature files (*.pt).",
        type=Path,
    )
    parser.add_argument(
        "vocoder_checkpoint_path",
        metavar="vocoder_checkpoint_path",
        help="vocoder checkpoint path",
        type = Path,
    )
    parser.add_argument(
        "segmenter_checkpoint_path",
        metavar="segmenter-checkpoint-path",
        help="path to the segmenter model checkpoint.",
        type=Path,
    )
    parser.add_argument(
        "knnvc",
        metavar="knnvc",
        choices=["knnvc", "rhythm-only", "knnvc-only"],
        type=str,
    )
    parser.add_argument("out_dir", metavar="out-dir", type=Path, help="path to the output directory.")
    args = parser.parse_args()
    convert(
        args.src_speaker_id,
        args.tgt_speaker_id,
        args.model_class,
        args.model_type,
        args.src_feats_dir,
        args.tgt_feats_dir,
        args.vocoder_checkpoint_path,
        args.segmenter_checkpoint_path,
        args.knnvc,
        args.out_dir,
    )