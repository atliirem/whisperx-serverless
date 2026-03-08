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
        except:
            pass

    if tracks is None and hasattr(raw, 'speaker_diarization'):
        try:
            sd = raw.speaker_diarization
            if hasattr(sd, 'itertracks'):
                tracks = [(t.start, t.end, s) for t, _, s in sd.itertracks(yield_label=True)]
        except:
            pass

    if tracks is None:
        for attr_name in dir(raw):
            attr = getattr(raw, attr_name, None)
            if attr and hasattr(attr, 'itertracks'):
                try:
                    tracks = [(t.start, t.end, s) for t, _, s in attr.itertracks(yield_label=True)]
                    break
                except:
                    pass

    if not tracks:
        raise ValueError(f"Diarization ciktisi islenemedi. Type: {type(raw)}")

    logger.info(f"Diarization: {len(tracks)} segment bulundu")
    return pd.DataFrame(tracks, columns=["start", "end", "speaker"])


def get_segment_pitch(audio_array, start_sec, end_sec, sr=16000):
    """Get median pitch for a specific time range using librosa pyin"""
    try:
        import librosa

        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        if end_sample > len(audio_array):
            end_sample = len(audio_array)
        if end_sample - start_sample < int(0.1 * sr):
            return 0

        chunk = audio_array[start_sample:end_sample].astype(np.float32)

        # librosa pyin - very accurate pitch detection
        f0, voiced_flag, voiced_probs = librosa.pyin(
            chunk, fmin=80, fmax=500, sr=sr,
            frame_length=2048, hop_length=512
        )

        # Only use voiced frames with high confidence
        valid = f0[voiced_flag & (voiced_probs > 0.5)]
        if len(valid) < 3:
            return 0

        return float(np.median(valid))
    except Exception as e:
        return 0


def analyze_speakers_pitch(audio_array, segments, sr=16000):
    """Analyze pitch for each speaker using multiple segments"""
    speaker_pitches = {}

    for seg in segments:
        sp = seg.get("speaker", "UNKNOWN")
        if sp == "UNKNOWN":
            continue

        duration = seg["end"] - seg["start"]
        if duration < 0.5:
            continue

        pitch = get_segment_pitch(audio_array, seg["start"], seg["end"], sr)
        if 80 < pitch < 500:
            if sp not in speaker_pitches:
                speaker_pitches[sp] = []
            speaker_pitches[sp].append(pitch)

    # Calculate robust median for each speaker
    result = {}
    for sp, pitches in speaker_pitches.items():
        if len(pitches) >= 3:
            result[sp] = float(np.median(pitches))
        elif len(pitches) > 0:
            result[sp] = float(np.mean(pitches))

    return result


def correct_segments_by_pitch(segments, audio_array, teacher_name, student_name, teacher_pitch, student_pitch, sr=16000):
    """Correct individual segments that are mislabeled based on pitch"""
    pitch_threshold = (teacher_pitch + student_pitch) / 2
    corrections = 0

    for i, seg in enumerate(segments):
        sp = seg.get("speaker", "UNKNOWN")
        if sp == "UNKNOWN":
            continue

        duration = seg["end"] - seg["start"]
        if duration < 0.3:
            continue

        seg_pitch = get_segment_pitch(audio_array, seg["start"], seg["end"], sr)
        if seg_pitch <= 0:
            continue

        # Check if pitch matches assigned speaker
        if sp == teacher_name and seg_pitch > pitch_threshold + 20:
            # Labeled as teacher but sounds like child
            segments[i]["speaker"] = student_name
            corrections += 1
        elif sp == student_name and seg_pitch < pitch_threshold - 20:
            # Labeled as student but sounds like adult
            segments[i]["speaker"] = teacher_name
            corrections += 1

    return segments, corrections


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
            payload = {
                "success": False,
                "error": "Transkripsiyon bos - konusma bulunamadi",
                "filename": filename,
                "file_size_mb": round(file_size_mb, 1),
                "processing_time_seconds": elapsed,
                "full_text": "",
                "speaker_text": ""
            }
            send_webhook(payload)
            return payload

        logger.info(f"Transkripsiyon: {len(result['segments'])} segment")

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
            logger.error("DIARIZE MODEL NONE!")

        raw_segments = result["segments"]

        # STEP 1: Pitch analysis for each speaker
        logger.info("Pitch analizi basliyor...")
        speaker_pitches = analyze_speakers_pitch(audio, raw_segments, sr=16000)
        for sp, pitch in speaker_pitches.items():
            logger.info(f"Speaker {sp}: median pitch = {round(pitch, 1)} Hz")

        # STEP 2: Assign TEACHER/STUDENT based on pitch
        speaker_map = {}
        unique_speakers = [sp for sp in set(seg.get("speaker", "UNKNOWN") for seg in raw_segments) if sp != "UNKNOWN"]

        if len(speaker_pitches) >= 2:
            sorted_by_pitch = sorted(speaker_pitches.items(), key=lambda x: x[1])
            teacher_sp = sorted_by_pitch[0][0]
            student_sp = sorted_by_pitch[-1][0]
            teacher_pitch = sorted_by_pitch[0][1]
            student_pitch = sorted_by_pitch[-1][1]

            speaker_map[teacher_sp] = "TEACHER"
            speaker_map[student_sp] = "STUDENT"
            logger.info(f"Pitch atamasi: TEACHER={teacher_sp} ({round(teacher_pitch,1)} Hz), STUDENT={student_sp} ({round(student_pitch,1)} Hz)")

            # STEP 3: Per-segment pitch correction
            if abs(teacher_pitch - student_pitch) > 30:
                logger.info("Segment bazli pitch duzeltme basliyor...")
                raw_segments, corrections = correct_segments_by_pitch(
                    raw_segments, audio, teacher_sp, student_sp,
                    teacher_pitch, student_pitch, sr=16000
                )
                logger.info(f"Pitch duzeltme: {corrections} segment degistirildi")
            else:
                logger.warning(f"Pitch farki cok kucuk ({round(abs(teacher_pitch - student_pitch),1)} Hz), segment duzeltme atlanıyor")
        else:
            # Fallback: most talking = TEACHER
            logger.warning("Pitch analizi yetersiz, sure bazli atama")
            speaker_durations = {}
            for seg in raw_segments:
                sp = seg.get("speaker", "UNKNOWN")
                dur = seg["end"] - seg["start"]
                speaker_durations[sp] = speaker_durations.get(sp, 0) + dur
            sorted_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
            for i, (sp, dur) in enumerate(sorted_speakers):
                speaker_map[sp] = "TEACHER" if i == 0 else "STUDENT"

        if "UNKNOWN" not in speaker_map:
            speaker_map["UNKNOWN"] = "UNKNOWN"

        logger.info(f"Final speaker map: {speaker_map}")

        # Build output
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
            "diarization_ok": diarization_ok,
            "speaker_pitches": {speaker_map.get(sp, sp): round(p, 1) for sp, p in speaker_pitches.items()}
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
