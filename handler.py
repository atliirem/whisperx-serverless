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


def download_file(url, max_retries=3):
    """Download file with retry logic"""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            r = req.get(url, timeout=300)
            r.raise_for_status()
            content = r.content
            tmp.write(content)
            tmp.close()
            size_mb = round(len(content) / (1024 * 1024), 1)

            if size_mb < 0.01:
                logger.warning(f"Indirme denemesi {attempt}/{max_retries}: Dosya cok kucuk ({size_mb} MB)")
                os.unlink(tmp.name)
                if attempt < max_retries:
                    time.sleep(3 * attempt)  # 3s, 6s, 9s bekle
                    continue
                else:
                    raise ValueError(f"Dosya {max_retries} denemede de bos indi ({size_mb} MB)")

            logger.info(f"Dosya indirildi: {size_mb} MB (deneme {attempt})")
            return tmp.name

        except Exception as e:
            last_error = e
            logger.warning(f"Indirme denemesi {attempt}/{max_retries} basarisiz: {e}")
            try:
                os.unlink(tmp.name)
            except:
                pass
            if attempt < max_retries:
                time.sleep(3 * attempt)

    raise last_error or ValueError("Dosya indirilemedi")


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
    try:
        import librosa
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        if end_sample > len(audio_array):
            end_sample = len(audio_array)
        if end_sample - start_sample < int(0.15 * sr):
            return 0

        chunk = audio_array[start_sample:end_sample].astype(np.float32)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            chunk, fmin=80, fmax=500, sr=sr,
            frame_length=2048, hop_length=512
        )
        valid = f0[voiced_flag & (voiced_probs > 0.6)]
        if len(valid) < 3:
            return 0
        return float(np.median(valid))
    except:
        return 0


def analyze_speakers(audio_array, segments, sr=16000):
    speakers = {}
    for seg in segments:
        sp = seg.get("speaker", "UNKNOWN")
        if sp == "UNKNOWN":
            continue
        dur = seg["end"] - seg["start"]
        if sp not in speakers:
            speakers[sp] = {"duration": 0, "count": 0, "pitches": []}
        speakers[sp]["duration"] += dur
        speakers[sp]["count"] += 1
        if dur >= 0.5:
            pitch = get_segment_pitch(audio_array, seg["start"], seg["end"], sr)
            if 80 < pitch < 500:
                speakers[sp]["pitches"].append(pitch)

    for sp in speakers:
        s = speakers[sp]
        s["avg_segment_length"] = s["duration"] / max(s["count"], 1)
        s["median_pitch"] = float(np.median(s["pitches"])) if len(s["pitches"]) >= 3 else 0
        s["pitch_count"] = len(s["pitches"])

    return speakers


def identify_teacher_student(speaker_stats):
    if len(speaker_stats) < 2:
        return None, None, {}

    speaker_list = list(speaker_stats.keys())

    # Duration = primary signal (teacher ALWAYS talks more)
    duration_sorted = sorted(speaker_list, key=lambda sp: speaker_stats[sp]["duration"], reverse=True)
    teacher = duration_sorted[0]
    student = duration_sorted[1]

    dur_ratio = speaker_stats[teacher]["duration"] / max(speaker_stats[student]["duration"], 0.1)
    logger.info(f"Duration: {teacher}={round(speaker_stats[teacher]['duration'],1)}s, "
                f"{student}={round(speaker_stats[student]['duration'],1)}s, ratio={round(dur_ratio,1)}x")

    # Pitch info for logging
    for sp in [teacher, student]:
        if speaker_stats[sp]["median_pitch"] > 0:
            logger.info(f"Pitch {sp}: {round(speaker_stats[sp]['median_pitch'],1)} Hz")

    confidence = "HIGH" if dur_ratio > 2 else ("MEDIUM" if dur_ratio > 1.3 else "LOW")
    logger.info(f"KARAR: TEACHER={teacher}, STUDENT={student}, Confidence={confidence}")

    info = {
        "teacher_duration": round(speaker_stats[teacher]["duration"], 1),
        "student_duration": round(speaker_stats[student]["duration"], 1),
        "duration_ratio": round(dur_ratio, 1),
        "teacher_pitch": round(speaker_stats[teacher]["median_pitch"], 1),
        "student_pitch": round(speaker_stats[student]["median_pitch"], 1),
        "confidence": confidence
    }
    return teacher, student, info


def correct_segments_by_pitch(segments, audio_array, teacher_name, student_name, teacher_pitch, student_pitch, sr=16000):
    if teacher_pitch <= 0 or student_pitch <= 0:
        return segments, 0

    pitch_diff = abs(student_pitch - teacher_pitch)
    if pitch_diff < 40:
        logger.info(f"Pitch farki kucuk ({round(pitch_diff,1)} Hz), duzeltme atlanıyor")
        return segments, 0

    midpoint = (teacher_pitch + student_pitch) / 2
    buffer = pitch_diff * 0.15

    corrections = 0
    for i, seg in enumerate(segments):
        sp = seg.get("speaker", "UNKNOWN")
        if sp == "UNKNOWN":
            continue
        duration = seg["end"] - seg["start"]
        if duration < 0.5:
            continue

        seg_pitch = get_segment_pitch(audio_array, seg["start"], seg["end"], sr)
        if seg_pitch <= 0:
            continue

        if sp == teacher_name and seg_pitch > midpoint + buffer:
            segments[i]["speaker"] = student_name
            corrections += 1
        elif sp == student_name and seg_pitch < midpoint - buffer:
            segments[i]["speaker"] = teacher_name
            corrections += 1

    return segments, corrections


def process_audio(audio, wav_path, language, min_speakers, max_speakers):
    """Core processing: transcribe + align + diarize + identify speakers"""
    
    result = model.transcribe(audio, language=language, batch_size=16)

    if not result.get("segments"):
        return None, "Transkripsiyon bos - konusma bulunamadi"

    logger.info(f"Transkripsiyon: {len(result['segments'])} segment")

    # Align
    try:
        align_model, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
        result = whisperx.align(result["segments"], align_model, metadata, audio, device=DEVICE)
        del align_model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Hizalama atlandi: {e}")

    # Diarize
    diarization_ok = False
    if diarize_model is not None:
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

    raw_segments = result["segments"]

    # Speaker identification
    speaker_stats = analyze_speakers(audio, raw_segments, sr=16000)
    for sp, stats in speaker_stats.items():
        logger.info(f"  {sp}: dur={round(stats['duration'],1)}s, pitch={round(stats['median_pitch'],1)} Hz")

    teacher_name, student_name, id_info = identify_teacher_student(speaker_stats)

    speaker_map = {}
    if teacher_name and student_name:
        speaker_map[teacher_name] = "TEACHER"
        speaker_map[student_name] = "STUDENT"

        tp = speaker_stats[teacher_name]["median_pitch"]
        sp = speaker_stats[student_name]["median_pitch"]
        if tp > 0 and sp > 0:
            raw_segments, corrections = correct_segments_by_pitch(
                raw_segments, audio, teacher_name, student_name, tp, sp, sr=16000
            )
            logger.info(f"Pitch duzeltme: {corrections} segment degistirildi")
    else:
        for seg in raw_segments:
            s = seg.get("speaker", "UNKNOWN")
            if s not in speaker_map:
                speaker_map[s] = "TEACHER"

    if "UNKNOWN" not in speaker_map:
        speaker_map["UNKNOWN"] = "UNKNOWN"

    return {
        "segments": raw_segments,
        "speaker_map": speaker_map,
        "diarization_ok": diarization_ok,
        "id_info": id_info
    }, None


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

        # Download with retry
        logger.info(f"Indiriliyor: {filename or file_url[:50]}")
        tmp_path = download_file(file_url, max_retries=3)
        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)

        wav_path = preprocess_audio(tmp_path)
        wav_size = os.path.getsize(wav_path) / (1024 * 1024)
        logger.info(f"WAV: {round(wav_size, 1)} MB")

        if wav_size < 0.01:
            payload = {
                "success": False,
                "error": "Ses dosyasi bos veya bozuk",
                "filename": filename,
                "file_size_mb": round(file_size_mb, 1),
                "processing_time_seconds": round(time.time() - start_time, 1)
            }
            send_webhook(payload)
            return payload

        audio = whisperx.load_audio(wav_path)

        # Process with retry: if first attempt gives empty result, try once more
        logger.info("Adim 1: Isleme basliyor...")
        proc_result, error = process_audio(audio, wav_path, language, min_speakers, max_speakers)

        if error or proc_result is None:
            logger.warning(f"Ilk deneme basarisiz: {error}. 5 saniye bekleyip tekrar deneniyor...")
            time.sleep(5)
            gc.collect()
            torch.cuda.empty_cache()
            audio = whisperx.load_audio(wav_path)
            proc_result, error = process_audio(audio, wav_path, language, min_speakers, max_speakers)

        if error or proc_result is None:
            elapsed = round(time.time() - start_time, 1)
            payload = {
                "success": False,
                "error": error or "Isleme basarisiz",
                "filename": filename,
                "file_size_mb": round(file_size_mb, 1),
                "processing_time_seconds": elapsed,
                "full_text": "", "speaker_text": ""
            }
            send_webhook(payload)
            return payload

        raw_segments = proc_result["segments"]
        speaker_map = proc_result["speaker_map"]
        diarization_ok = proc_result["diarization_ok"]
        id_info = proc_result["id_info"]

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

        # Final check: if speaker_text is empty, mark as failed
        if not speaker_text.strip():
            payload = {
                "success": False,
                "error": "Transkript bos - isleme basarili ama metin uretilmedi",
                "filename": filename,
                "file_size_mb": round(file_size_mb, 1),
                "processing_time_seconds": elapsed,
                "full_text": "", "speaker_text": ""
            }
            send_webhook(payload)
            return payload

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
            "identification_info": id_info
        }

        send_webhook(payload)
        logger.info(f"Tamamlandi! Sure: {elapsed}s | Segments: {len(segments)} | Diarization: {diarization_ok}")
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
