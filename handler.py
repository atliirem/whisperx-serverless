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
        diarize_model = whisperx.DiarizePipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        logger.info("Diarization modeli hazir (GPU)!")
    except Exception as e:
        logger.error(f"Diarization yuklenemedi: {e}")


def preprocess_audio(input_path):
    output_path = input_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path, "-y"],
            capture_output=True, timeout=120
        )
        return output_path
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

    try:
        if not file_url:
            return {"success": False, "error": "file_url gerekli"}

        logger.info(f"Indiriliyor: {filename or file_url[:50]}")
        tmp_path = download_file(file_url)
        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)

        wav_path = preprocess_audio(tmp_path)

        logger.info("Adim 1/3: Transkripsiyon...")
        audio = whisperx.load_audio(wav_path)
        result = model.transcribe(audio, language=language, batch_size=16)

        logger.info("Adim 2/3: Kelime hizalama...")
        try:
            align_model, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
            result = whisperx.align(result["segments"], align_model, metadata, audio, device=DEVICE)
            del align_model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Hizalama atlandi: {e}")

        if do_diarize and diarize_model is not None:
            logger.info("Adim 3/3: Konusmaci ayristirma...")
            try:
                diarize_segments = diarize_model(wav_path, min_speakers=min_speakers, max_speakers=max_speakers)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                logger.info("Diarization basarili!")
            except Exception as e:
                logger.warning(f"Konusmaci ayristirma hatasi: {e}")

        # Rename speakers: most talking = TEACHER
        raw_segments = result["segments"]
        speaker_durations = {}
        for seg in raw_segments:
            sp = seg.get("speaker", "UNKNOWN")
            dur = seg["end"] - seg["start"]
            speaker_durations[sp] = speaker_durations.get(sp, 0) + dur
        sorted_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
        speaker_map = {}
        for i, (sp, dur) in enumerate(sorted_speakers):
            if i == 0:
                speaker_map[sp] = "TEACHER"
            else:
                speaker_map[sp] = "STUDENT"

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
            "speaker_text": speaker_text
        }

        if webhook_url:
            try:
                req.post(webhook_url, json=payload, timeout=30)
                logger.info(f"Webhook gonderildi! Sure: {elapsed}s")
            except Exception as e:
                logger.error(f"Webhook hatasi: {e}")

        logger.info(f"Tamamlandi! Sure: {elapsed}s")
        return payload

    except Exception as e:
        elapsed = round(time.time() - start_time, 1)
        logger.error(f"Hata: {e}")
        error_payload = {
            "success": False,
            "filename": filename,
            "error": str(e),
            "processing_time_seconds": elapsed
        }
        if webhook_url:
            try:
                req.post(webhook_url, json=error_payload, timeout=30)
            except:
                pass
        return error_payload

    finally:
        for f in [tmp_path, wav_path]:
            try:
                if f:
                    os.unlink(f)
            except:
                pass
        gc.collect()
        torch.cuda.empty_cache()


runpod.serverless.start({"handler": handler})
