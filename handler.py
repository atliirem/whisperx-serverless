import os, gc, time, tempfile, subprocess, logging
import requests as req
import runpod

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
    """Extract speaker segments from pyannote output, handling both v3 and v4 API"""
    import pandas as pd
    
    # Try different pyannote output formats
    tracks = None
    
    # Method 1: Direct itertracks (pyannote 3.x)
    if hasattr(raw, 'itertracks'):
        try:
            tracks = [(t.start, t.end, s) for t, _, s in raw.itertracks(yield_label=True)]
            logger.info(f"Diarization: itertracks ile {len(tracks)} segment bulundu")
        except Exception as e:
            logger.warning(f"itertracks hatasi: {e}")
    
    # Method 2: speaker_diarization attribute (pyannote 4.x)
    if tracks is None and hasattr(raw, 'speaker_diarization'):
        try:
            sd = raw.speaker_diarization
            if hasattr(sd, 'itertracks'):
                tracks = [(t.start, t.end, s) for t, _, s in sd.itertracks(yield_label=True)]
                logger.info(f"Diarization: speaker_diarization.itertracks ile {len(tracks)} segment bulundu")
        except Exception as e:
            logger.warning(f"speaker_diarization.itertracks hatasi: {e}")
    
    # Method 3: Try to iterate raw as annotation
    if tracks is None:
        try:
            # Some versions return annotation directly
            for attr_name in dir(raw):
                attr = getattr(raw, attr_name)
                if hasattr(attr, 'itertracks'):
                    tracks = [(t.start, t.end, s) for t, _, s in attr.itertracks(yield_label=True)]
                    logger.info(f"Diarization: {attr_name}.itertracks ile {len(tracks)} segment bulundu")
                    break
        except Exception as e:
            logger.warning(f"Fallback itertracks hatasi: {e}")
    
    # Method 4: Log all attributes to debug
    if tracks is None:
        logger.error(f"Diarization output type: {type(raw)}")
        logger.error(f"Diarization output dir: {[a for a in dir(raw) if not a.startswith('_')]}")
        try:
            logger.error(f"Diarization output str: {str(raw)[:500]}")
        except:
            pass
        raise ValueError(f"Diarization ciktisi isle namadi. Type: {type(raw)}")
    
    if not tracks:
        raise ValueError("Diarization hicbir konusmaci bulamadi")
    
    return pd.DataFrame(tracks, columns=["start", "end", "speaker"])


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

        # Rename speakers: most talking = TEACHER
        raw_segments = result["segments"]
        speaker_durations = {}
        for seg in raw_segments:
            sp = seg.get("speaker", "UNKNOWN")
            dur = seg["end"] - seg["start"]
            speaker_durations[sp] = speaker_durations.get(sp, 0) + dur

        logger.info(f"Speaker durations: {speaker_durations}")

        sorted_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
        speaker_map = {}
        for i, (sp, dur) in enumerate(sorted_speakers):
            if i == 0:
                speaker_map[sp] = "TEACHER"
            else:
                speaker_map[sp] = "STUDENT"

        logger.info(f"Speaker map: {speaker_map}")

        # Post-processing: short responses after teacher questions = STUDENT
        teacher_name = sorted_speakers[0][0] if sorted_speakers else None
        student_name = sorted_speakers[1][0] if len(sorted_speakers) > 1 else None

        if teacher_name and student_name:
            for i in range(1, len(raw_segments)):
                prev = raw_segments[i-1]
                curr = raw_segments[i]
                curr_duration = curr["end"] - curr["start"]
                curr_speaker = curr.get("speaker", "UNKNOWN")
                prev_speaker = prev.get("speaker", "UNKNOWN")
                prev_text = prev.get("text", "").strip()
                curr_text = curr.get("text", "").strip()

                gap = curr["start"] - prev["end"]
                is_question = prev_text.endswith("?")
                is_prompt = any(w in prev_text.lower() for w in [
                    "say it", "repeat", "again", "what", "choose",
                    "yes or no", "yes no", "is it", "do you", "can you",
                    "tell me", "let's say", "one more"
                ])
                is_short = len(curr_text.split()) <= 8 and curr_duration < 6
                is_echo = curr_text.lower().strip(".") in prev_text.lower()

                if (curr_speaker == teacher_name and prev_speaker == teacher_name
                        and gap < 3
                        and is_short
                        and (is_question or is_prompt or is_echo)):
                    raw_segments[i]["speaker"] = student_name
                    logger.info(f"Relabeled to STUDENT: {curr_text}")

        logger.info("Post-processing tamamlandi")

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
