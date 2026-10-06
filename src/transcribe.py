"""
ASR Transcription module using Whisper.
High-performance multiprocessing pipeline transcribing audio files across multiple CPU cores,
extracting word timestamps, pauses, and speech temporal features.
"""

import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    TRAIN_CSV,
    TEST_CSV,
    TRANSCRIPTS_TRAIN_FILE,
    TRANSCRIPTS_TEST_FILE,
    WHISPER_MODEL_NAME,
    AUDIO_SAMPLE_RATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global worker pipeline
_worker_pipe = None


def init_worker(model_id: str):
    global _worker_pipe
    torch.set_num_threads(2)
    device = "cpu"
    torch_dtype = torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    _worker_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
        chunk_length_s=30,
        stride_length_s=(4, 2),
    )


def process_single_audio_task(args: Tuple[str, str]) -> Tuple[str, Dict[str, Any]]:
    fn, audio_path = args
    global _worker_pipe

    if not os.path.exists(audio_path):
        return fn, {"text": "", "word_count": 0, "error": "file_not_found"}

    try:
        data, sr = sf.read(audio_path)
        if sr != AUDIO_SAMPLE_RATE:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
            sr = AUDIO_SAMPLE_RATE
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        total_duration = len(data) / float(sr)

        # Fast check: If audio is high-frequency static noise (ZCR > 0.35)
        zc = np.sum((data[:-1] * data[1:]) < 0) / float(len(data))
        if zc > 0.35:
            return fn, {
                "text": "",
                "word_count": 0,
                "total_duration": total_duration,
                "speech_duration": 0.0,
                "speaking_rate": 0.0,
                "articulation_rate": 0.0,
                "num_pauses": 0,
                "num_long_pauses": 0,
                "mean_pause": 0.0,
                "max_pause": 0.0,
                "silence_ratio": 1.0,
                "is_noise": True,
            }

        # Transcribe with Whisper
        result = _worker_pipe(
            {"raw": data, "sampling_rate": sr},
            return_timestamps=True,
        )

        text = result.get("text", "").strip()
        chunks = result.get("chunks", [])

        speech_duration = 0.0
        pauses = []
        last_end = 0.0

        for chunk in chunks:
            ts = chunk.get("timestamp", (None, None))
            if ts and ts[0] is not None and ts[1] is not None:
                start, end = ts
                seg_dur = max(0.0, end - start)
                speech_duration += seg_dur
                if start > last_end and last_end > 0.0:
                    pauses.append(start - last_end)
                last_end = max(last_end, end)

        words = text.split()
        word_count = len(words)
        speaking_rate = (word_count / total_duration) if total_duration > 0 else 0.0
        articulation_rate = (word_count / speech_duration) if speech_duration > 0 else 0.0

        long_pauses = [p for p in pauses if p >= 0.5]
        mean_pause = float(np.mean(pauses)) if pauses else 0.0
        max_pause = float(np.max(pauses)) if pauses else 0.0
        silence_ratio = max(0.0, total_duration - speech_duration) / total_duration if total_duration > 0 else 0.0

        return fn, {
            "text": text,
            "word_count": word_count,
            "total_duration": total_duration,
            "speech_duration": speech_duration,
            "speaking_rate": speaking_rate,
            "articulation_rate": articulation_rate,
            "num_pauses": len(pauses),
            "num_long_pauses": len(long_pauses),
            "mean_pause": mean_pause,
            "max_pause": max_pause,
            "silence_ratio": silence_ratio,
        }
    except Exception as e:
        return fn, {
            "text": "",
            "word_count": 0,
            "total_duration": 0.0,
            "speech_duration": 0.0,
            "speaking_rate": 0.0,
            "articulation_rate": 0.0,
            "num_pauses": 0,
            "num_long_pauses": 0,
            "mean_pause": 0.0,
            "max_pause": 0.0,
            "silence_ratio": 1.0,
            "error": str(e),
        }


def transcribe_dataset(
    audio_dir: Path, csv_path: Path, output_json: Path, num_workers: int = 4
):
    transcripts = {}
    if output_json.exists():
        try:
            with open(output_json, "r") as f:
                transcripts = json.load(f)
            logger.info(f"Loaded {len(transcripts)} existing transcripts from {output_json}")
        except Exception as e:
            logger.warning(f"Could not load existing transcripts: {e}")

    df = pd.read_csv(csv_path)
    remaining_tasks = [
        (fn, str(audio_dir / fn)) for fn in df["filename"] if fn not in transcripts
    ]

    if not remaining_tasks:
        logger.info(f"All {len(df)} files already transcribed in {output_json}!")
        return transcripts

    logger.info(
        f"Transcribing {len(remaining_tasks)} remaining files with {num_workers} worker processes..."
    )

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=num_workers, initializer=init_worker, initargs=(WHISPER_MODEL_NAME,)
    ) as pool:
        saved_counter = 0
        pbar = tqdm(total=len(remaining_tasks), desc=f"Transcribing {csv_path.stem}")
        for fn, res in pool.imap_unordered(process_single_audio_task, remaining_tasks, chunksize=1):
            transcripts[fn] = res
            saved_counter += 1
            pbar.update(1)

            if saved_counter % 25 == 0:
                output_json.parent.mkdir(parents=True, exist_ok=True)
                with open(output_json, "w") as f:
                    json.dump(transcripts, f, indent=2)
        pbar.close()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(transcripts, f, indent=2)

    logger.info(f"Saved all {len(transcripts)} transcripts to {output_json}")
    return transcripts


def main():
    logger.info("Starting Multi-Process ASR transcription...")
    num_workers = min(4, os.cpu_count() or 4)
    logger.info(f"Using {num_workers} parallel workers on {os.cpu_count()} CPU cores")
    transcribe_dataset(TRAIN_DIR, TRAIN_CSV, TRANSCRIPTS_TRAIN_FILE, num_workers=num_workers)
    transcribe_dataset(TEST_DIR, TEST_CSV, TRANSCRIPTS_TEST_FILE, num_workers=num_workers)
    logger.info("Transcription completed successfully!")


if __name__ == "__main__":
    main()
