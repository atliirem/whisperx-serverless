import os, gc, time, tempfile, subprocess, logging
import requests as req
import runpod
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisperx-serverless")

DEVICE = "cuda"
COMPUTE_TYPE = "float16"
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

import whisperx
import torch

logger.info("Model yukleniyor...")
model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
logger.info("Model hazir!")

diarize_model = None
if HF_TOKEN:
    logger.info("Diarization modeli yukleniyor...")
    try:
        from pyannote.audio import Pipeline
        diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)
        diarize_model.to(torch.device(DEVICE))
        logger.info("Diarization modeli hazir (pyannote GPU)!")
    except Exception as e:
        logger.error(f"Diarization yuklenemedi: {e}")
else:
    logger.warning("HF_TOKEN yok, diarization calismayacak!")


def preprocess_audio(input_path):
    output_path = input_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path, "-y"],
            capture_output=True, timeout=120
        )
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return input_path
    except:
        return input_path


def download_file(url):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    r = req.get(url, timeout=300)
    r.raise_for_status()
    tmp.write(r.content)
    tmp.close()
    logger.info(f"Dosya indirildi: {round(len(r.content)/(1024*1024),1)} MB")
    return tmp.name


def extract_diarization(raw):
    import pandas as pd
    tracks = None

    if hasattr(raw, 'itertracks'):
        try:
            tracks = [(t.start, t.end, s) for t, _, s in raw.itertracks(yield_label=True)]
            logger.info(f"Diarization: itertracks ile {len(tracks)} segment bulundu")
        except Exception as e:
            logger.warning(f"itertracks hatasi: {e}")

    if tracks is None and hasattr(raw, 'speaker_diarization'):
        try:
            sd = raw.speaker_diarization
            if hasattr(sd, 'itertracks'):
                tracks = [(t.start, t.end, s) for t, _, s in sd.itertracks(yield_label=True)]
                logger.info(f"Diarization: speaker_diarization.itertracks ile {len(tracks)} segment bulundu")
        except Exception as e:
            logger.warning(f"speaker_diarization.itertracks hatasi: {e}")

    if tracks is None:
        try:
            for attr_name in dir(raw):
                attr = getattr(raw, attr_name)
                if hasattr(attr, 'itertracks'):
                    tracks = [(t.start, t.end, s) for t, _, s in attr.itertracks(yield_label=True)]
                    logger.info(f"Diarization: {attr_name}.itertracks ile {len(tracks)} segment bulundu")
                    break
        except Exception as e:
            logger.warning(f"Fallback itertracks hatasi: {e}")

    if tracks is None:
        logger.error(f"Diarization output type: {type(raw)}")
        raise ValueError(f"Diarization ciktisi islenemedi. Type: {type(raw)}")

    if not tracks:
        raise ValueError("Diarization hicbir konusmaci bulamadi")

    return pd.DataFrame(tracks, columns=["start", "end", "speaker"])


def get_pitch(audio_array, sr=16000):
    """Calculate fundamental frequency (F0) using autocorrelation"""
    # Only analyze segments with enough energy (voice activity)
    if len(audio_array) < sr * 0.05:  # minimum 50ms
        return 0

    # Normalize
    audio_array = audio_array.astype(np.float64)
    if np.max(np.abs(audio_array)) == 0:
        return 0
    audio_array = audio_array / np.max(np.abs(audio_array))

    # Use autocorrelation for pitch detection
    # F0 range: 80 Hz (deep male) to 500 Hz (child)
    min_lag = int(sr / 500)  # 500 Hz -> highest pitch
    max_lag = int(sr / 80)   # 80 Hz -> lowest pitch

    if max_lag >= len(audio_array):
        return 0

    # Autocorrelation
    corr = np.correlate(audio_array, audio_array, mode='full')
    corr = corr[len(corr)//2:]  # Take positive lags only

    if max_lag >= len(corr):
        return 0

    # Find peak in valid range
    search_range = corr[min_lag:max_lag]
    if len(search_range) == 0:
        return 0

    peak_idx = np.argmax(search_range) + min_lag
    if corr[peak_idx] <= 0:
        return 0

    f0 = sr / peak_idx
    return f0


def analyze_speaker_pitch(audio_array, segments, speaker_name, sr=16000):
    """Calculate average pitch for a specific speaker's segments"""
    pitches = []

    for seg in segments:
        if seg.get("speaker") != speaker_name:
            continue

        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)

        if end_sample > len(audio_array):
            end_sample = len(audio_array)
        if start_sample >= end_sample:
            continue

        chunk = audio_array[start_sample:end_sample]

        # Split into 200ms windows for better pitch estimation
        window_size = int(0.2 * sr)
        for j in range(0, len(chunk) - window_size, window_size):
            window = chunk[j:j+window_size]
            f0 = get_pitch(window, sr)
            if 80 < f0 < 500:  # Valid voice range
                pitches.append(f0)

    if not pitches:
        return 0

    # Use median to be robust against outliers
    return float(np.median(pitches))


def handler(job):
    job_input = job["input"]
    start_time = time.time()

    file_url = job_input.get("file_url", "")
    language = job_input.get("language", "en")
    do_diarize = str(job_input.get("diarize", "true")).lower() in ("true", "1", "yes")
    min_speakers = int(job_input.get("min_speakers", 2))
    max_speakers = int(job_input.get("max_speakers", 2))
    webhook_url = job_input.get("webhook_url", "")
    filename = job_input.get("filename", "")

    tmp_path = None
    wav_path = None

    def send_webhook(payload):
        if webhook_url:
            try:
                req.post(webhook_url, json=payload, timeout=30)
                logger.info(f"Webhook gonderildi: success={payload.get('success')}")
            except Exception as e:
                logger.error(f"Webhook hatasi: {e}")

    try:
        if not file_url:
            payload = {"success": False, "error": "file_url gerekli", "filename": filename}
            send_webhook(payload)
            return payload

        logger.info(f"Indiriliyor: {filename or file_url[:50]}")
        tmp_path = download_file(file_url)
        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)

        wav_path = preprocess_audio(tmp_path)
        wav_size = os.path.getsize(wav_path) / (1024 * 1024)
        logger.info(f"WAV dosya boyutu: {round(wav_size, 1)} MB")

        if wav_size < 0.01:
            payload = {
                "success": False,
                "error": "Ses dosyasi bos veya bozuk (< 10KB)",
                "filename": filename,
                "file_size_mb": round(file_size_mb, 1),
                "processing_time_seconds": round(time.time() - start_time, 1)
            }
            send_webhook(payload)
            return payload

        logger.info("Adim 1/3: Transkripsiyon...")
        audio = whisperx.load_audio(wav_path)
        result = model.transcribe(audio, language=language, batch_size=16)

        if not result.get("segments"):
            elapsed = round(time.time() - start_time, 1)
            logger.error(f"Transkripsiyon bos dondu! Dosya: {filename}")
            payload = {
                "success": False,
                "error": "Transkripsiyon bos - ses dosyasinda konusma bulunamadi",
                "filename": filename,
                "file_size_mb": round(file_size_mb, 1),
                "processing_time_seconds": elapsed,
                "full_text": "",
                "speaker_text": ""
            }
            send_webhook(payload)
            return payload

        logger.info(f"Transkripsiyon tamamlandi: {len(result['segments'])} segment")

        logger.info("Adim 2/3: Kelime hizalama...")
        try:
            align_model, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
            result = whisperx.align(result["segments"], align_model, metadata, audio, device=DEVICE)
            del align_model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Hizalama atlandi: {e}")

        diarization_ok = False
        if do_diarize and diarize_model is not None:
            logger.info("Adim 3/3: Konusmaci ayristirma...")
            try:
                import torchaudio
                waveform, sr = torchaudio.load(wav_path)
                raw = diarize_model(
                    {"waveform": waveform.to(DEVICE), "sample_rate": sr},
                    min_speakers=min_speakers, max_speakers=max_speakers
                )
                diarize_df = extract_diarization(raw)
                result = whisperx.assign_word_speakers(diarize_df, result)
                diarization_ok = True
                logger.info("Diarization basarili!")
            except Exception as e:
                logger.error(f"Diarization hatasi: {e}")
        elif diarize_model is None:
            logger.error("DIARIZE MODEL NONE - yuklenmemis!")

        # Identify speakers using PITCH analysis
        raw_segments = result["segments"]
        unique_speakers = list(set(seg.get("speaker", "UNKNOWN") for seg in raw_segments))
        logger.info(f"Unique speakers: {unique_speakers}")

        speaker_map = {}

        if len(unique_speakers) >= 2 and diarization_ok:
            # Analyze pitch for each speaker
            speaker_pitches = {}
            for sp in unique_speakers:
                if sp == "UNKNOWN":
                    continue
                avg_pitch = analyze_speaker_pitch(audio, raw_segments, sp, sr=16000)
                speaker_pitches[sp] = avg_pitch
                logger.info(f"Speaker {sp} ortalama pitch: {round(avg_pitch, 1)} Hz")

            # Higher pitch = child (STUDENT), lower pitch = adult (TEACHER)
            valid_pitches = {sp: p for sp, p in speaker_pitches.items() if p > 0}

            if len(valid_pitches) >= 2:
                sorted_by_pitch = sorted(valid_pitches.items(), key=lambda x: x[1])
                # Lowest pitch = TEACHER (adult)
                # Highest pitch = STUDENT (child)
                speaker_map[sorted_by_pitch[0][0]] = "TEACHER"
                speaker_map[sorted_by_pitch[-1][0]] = "STUDENT"
                logger.info(f"Pitch-based assignment: TEACHER={sorted_by_pitch[0][0]} ({round(sorted_by_pitch[0][1],1)} Hz), STUDENT={sorted_by_pitch[-1][0]} ({round(sorted_by_pitch[-1][1],1)} Hz)")
            else:
                # Fallback: most talking = TEACHER
                logger.warning("Pitch analizi basarisiz, sure bazli atama yapiliyor")
                speaker_durations = {}
                for seg in raw_segments:
                    sp = seg.get("speaker", "UNKNOWN")
                    dur = seg["end"] - seg["start"]
                    speaker_durations[sp] = speaker_durations.get(sp, 0) + dur
                sorted_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
                for i, (sp, dur) in enumerate(sorted_speakers):
                    speaker_map[sp] = "TEACHER" if i == 0 else "STUDENT"
        else:
            # No diarization or single speaker - fallback to duration
            speaker_durations = {}
            for seg in raw_segments:
                sp = seg.get("speaker", "UNKNOWN")
                dur = seg["end"] - seg["start"]
                speaker_durations[sp] = speaker_durations.get(sp, 0) + dur
            sorted_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
            for i, (sp, dur) in enumerate(sorted_speakers):
                speaker_map[sp] = "TEACHER" if i == 0 else "STUDENT"

        # Ensure UNKNOWN maps to something
        if "UNKNOWN" not in speaker_map:
            speaker_map["UNKNOWN"] = "UNKNOWN"

        logger.info(f"Final speaker map: {speaker_map}")

        segments = []
        for seg in raw_segments:
            segments.append({
                "speaker": speaker_map.get(seg.get("speaker", "UNKNOWN"), "UNKNOWN"),
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip()
            })

        full_text = "\n".join(
            "[" + s["speaker"] + "] (" + str(s["start"]) + "s - " + str(s["end"]) + "s): " + s["text"]
            for s in segments
        )
        speaker_text = "\n".join(
            "[" + s["speaker"] + "]: " + s["text"]
            for s in segments
        )
        speakers = list(set(s["speaker"] for s in segments))
        elapsed = round(time.time() - start_time, 1)

        payload = {
            "success": True,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "language": language,
            "processing_time_seconds": elapsed,
            "file_size_mb": round(file_size_mb, 1),
            "speakers_found": len(speakers),
            "speakers": speakers,
            "segment_count": len(segments),
            "full_text": full_text,
            "speaker_text": speaker_text,
            "diarization_ok": diarization_ok
        }

        send_webhook(payload)
        logger.info(f"Tamamlandi! Sure: {elapsed}s | Diarization: {diarization_ok}")
        return payload

    except Exception as e:
        elapsed = round(time.time() - start_time, 1)
        logger.error(f"Kritik hata: {e}", exc_info=True)
        payload = {
            "success": False,
            "filename": filename,
            "error": str(e),
            "processing_time_seconds": elapsed
        }
        send_webhook(payload)
        return payload

    finally:
        for f in [tmp_path, wav_path]:
            try:
                if f and os.path.exists(f):
                    os.unlink(f)
            except:
                pass
        gc.collect()
        torch.cuda.empty_cache()


runpod.serverless.start({"handler": handler})
