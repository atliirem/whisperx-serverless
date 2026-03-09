import os, gc, time, tempfile, subprocess, logging
import requests as req
import runpod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisperx-serverless")

DEVICE = "cuda"
COMPUTE_TYPE = "float16"
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Default prompt for English lessons with young students
DEFAULT_PROMPT = (
    "This is an online English lesson between a teacher and a young student aged 4 to 15. "
    "The teacher speaks clearly and slowly. The student is learning English and may make mistakes. "
    "Common words and phrases: hello, goodbye, good morning, good afternoon, good evening, good night, "
    "how are you, I am fine, thank you, yes, no, okay, please, sorry, "
    "colors: red, blue, yellow, green, purple, orange, pink, brown, black, white, "
    "animals: cat, dog, fish, bird, monkey, giraffe, zebra, elephant, lion, tiger, shark, dolphin, turtle, "
    "school items: pencil, pen, eraser, ruler, book, bag, school bag, pencil case, desk, chair, window, "
    "numbers: one, two, three, four, five, six, seven, eight, nine, ten, "
    "family: mother, father, sister, brother, baby, "
    "food: apple, banana, pizza, hamburger, chicken, rice, bread, milk, water, juice, "
    "places: school, home, park, zoo, beach, playground, sea world, "
    "verbs: like, want, have, see, eat, drink, play, swim, run, read, write, draw, sing, dance, "
    "my name is, it is a, I like, I have, I want, do you like, what is this, where is, "
    "very good, good job, well done, excellent, repeat after me, say it again, one more time."
)

import whisperx
import torch

logger.info("[INIT] Model yukleniyor...")
model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
logger.info("[INIT] Model hazir!")

diarize_model = None
if HF_TOKEN:
    logger.info("[INIT] Diarization modeli yukleniyor...")
    try:
        from pyannote.audio import Pipeline
        diarize_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=HF_TOKEN
        )
        diarize_model.to(torch.device(DEVICE))
        logger.info("[INIT] Diarization modeli hazir!")
    except Exception as e:
        logger.error(f"[INIT] Diarization yuklenemedi: {e}")
else:
    logger.warning("[INIT] HF_TOKEN yok!")


def preprocess_audio(input_path):
    output_path = input_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1",
             "-f", "wav", output_path, "-y"],
            capture_output=True, timeout=300
        )
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except:
        pass
    return input_path


def download_file(url, max_retries=3):
    import gdown

    gdrive_id = None
    if "drive.google.com" in url:
        if "id=" in url:
            gdrive_id = url.split("id=")[-1].split("&")[0]
        elif "/d/" in url:
            gdrive_id = url.split("/d/")[1].split("/")[0]

    for attempt in range(1, max_retries + 1):
        tmp = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
        tmp.close()
        try:
            if gdrive_id:
                gdown_url = f"https://drive.google.com/uc?id={gdrive_id}"
                logger.info(f"[DL] gdown deneme {attempt}: {gdrive_id}")
                gdown.download(gdown_url, tmp.name, quiet=True, fuzzy=True)
            else:
                r = req.get(url, timeout=300)
                r.raise_for_status()
                with open(tmp.name, "wb") as f:
                    f.write(r.content)

            size_mb = round(os.path.getsize(tmp.name) / (1024 * 1024), 1)

            if size_mb < 0.5:
                with open(tmp.name, "rb") as f:
                    header = f.read(200)
                if b"<!DOCTYPE" in header or b"<html" in header:
                    logger.warning(f"[DL] Deneme {attempt}: HTML sayfasi geldi")
                    os.unlink(tmp.name)
                    if attempt < max_retries:
                        time.sleep(5 * attempt)
                        continue
                    raise ValueError("Google Drive onay sayfasi asilamadi")

            if size_mb < 0.01:
                logger.warning(f"[DL] Deneme {attempt}: Bos dosya ({size_mb} MB)")
                os.unlink(tmp.name)
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
                raise ValueError(f"{max_retries} denemede bos indi")

            logger.info(f"[DL] Basarili: {size_mb} MB (deneme {attempt})")
            return tmp.name

        except Exception as e:
            logger.warning(f"[DL] Deneme {attempt} hata: {e}")
            try:
                os.unlink(tmp.name)
            except:
                pass
            if attempt < max_retries:
                time.sleep(5 * attempt)
            else:
                raise


def extract_diarization(raw):
    import pandas as pd

    if hasattr(raw, 'itertracks'):
        try:
            tracks = [(t.start, t.end, s) for t, _, s in raw.itertracks(yield_label=True)]
            if tracks:
                logger.info(f"[DIAR] itertracks: {len(tracks)} segment")
                return pd.DataFrame(tracks, columns=["start", "end", "speaker"])
        except:
            pass

    if hasattr(raw, 'speaker_diarization'):
        try:
            sd = raw.speaker_diarization
            if hasattr(sd, 'itertracks'):
                tracks = [(t.start, t.end, s) for t, _, s in sd.itertracks(yield_label=True)]
                if tracks:
                    logger.info(f"[DIAR] speaker_diarization: {len(tracks)} segment")
                    return pd.DataFrame(tracks, columns=["start", "end", "speaker"])
        except:
            pass

    for name in dir(raw):
        obj = getattr(raw, name, None)
        if obj and hasattr(obj, 'itertracks'):
            try:
                tracks = [(t.start, t.end, s) for t, _, s in obj.itertracks(yield_label=True)]
                if tracks:
                    logger.info(f"[DIAR] {name}: {len(tracks)} segment")
                    return pd.DataFrame(tracks, columns=["start", "end", "speaker"])
            except:
                pass

    raise ValueError(f"Diarization parse edilemedi: {type(raw)}")


def handler(job):
    inp = job["input"]
    t0 = time.time()

    file_url = inp.get("file_url", "")
    language = inp.get("language", "en")
    do_diarize = str(inp.get("diarize", "true")).lower() in ("true", "1", "yes")
    min_sp = int(inp.get("min_speakers", 2))
    max_sp = int(inp.get("max_speakers", 2))
    webhook_url = inp.get("webhook_url", "")
    filename = inp.get("filename", "")
    prompt = inp.get("prompt", DEFAULT_PROMPT)

    tmp_path = None
    wav_path = None

    def webhook(payload):
        if webhook_url:
            try:
                req.post(webhook_url, json=payload, timeout=30)
            except:
                pass

    def fail(error):
        p = {
            "success": False, "error": error, "filename": filename,
            "processing_time_seconds": round(time.time() - t0, 1),
            "full_text": "", "speaker_text": ""
        }
        webhook(p)
        return p

    try:
        if not file_url:
            return fail("file_url gerekli")

        logger.info(f"[JOB] Basliyor: {filename}")

        # 1. DOWNLOAD
        tmp_path = download_file(file_url)
        file_size_mb = round(os.path.getsize(tmp_path) / (1024 * 1024), 1)

        wav_path = preprocess_audio(tmp_path)
        wav_mb = round(os.path.getsize(wav_path) / (1024 * 1024), 1)
        logger.info(f"[JOB] WAV: {wav_mb} MB")

        if wav_mb < 0.01:
            return fail("Ses dosyasi bos veya bozuk")

        # 2. TRANSCRIBE with initial_prompt
        logger.info("[1/3] Transkripsiyon...")
        audio = whisperx.load_audio(wav_path)
        result = model.transcribe(
            audio,
            language=language,
            batch_size=16,
            initial_prompt=prompt
        )

        if not result.get("segments"):
            return fail("Transkripsiyon bos")

        logger.info(f"[1/3] {len(result['segments'])} segment")

        # 3. ALIGN
        logger.info("[2/3] Hizalama...")
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=language, device=DEVICE
            )
            result = whisperx.align(
                result["segments"], align_model, metadata, audio, device=DEVICE
            )
            del align_model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"[2/3] Hizalama atlandi: {e}")

        # 4. DIARIZE
        diarization_ok = False
        if do_diarize and diarize_model is not None:
            logger.info("[3/3] Diarization...")
            try:
                import torchaudio
                waveform, sr = torchaudio.load(wav_path)
                raw = diarize_model(
                    {"waveform": waveform.to(DEVICE), "sample_rate": sr},
                    min_speakers=min_sp, max_speakers=max_sp
                )
                diarize_df = extract_diarization(raw)
                result = whisperx.assign_word_speakers(diarize_df, result)
                diarization_ok = True
                logger.info("[3/3] Diarization basarili!")
            except Exception as e:
                logger.error(f"[3/3] Diarization hatasi: {e}")

        # 5. SPEAKER MAP (duration-based)
        segs = result["segments"]
        durations = {}
        for s in segs:
            sp = s.get("speaker", "UNKNOWN")
            durations[sp] = durations.get(sp, 0) + (s["end"] - s["start"])

        ranked = sorted(durations.items(), key=lambda x: x[1], reverse=True)
        sp_map = {}
        for i, (sp, dur) in enumerate(ranked):
            sp_map[sp] = "TEACHER" if i == 0 else "STUDENT"
        sp_map.setdefault("UNKNOWN", "UNKNOWN")

        logger.info(f"[MAP] {sp_map}")

        # 6. BUILD OUTPUT
        out = []
        for s in segs:
            out.append({
                "speaker": sp_map.get(s.get("speaker", "UNKNOWN"), "UNKNOWN"),
                "start": round(s["start"], 2),
                "end": round(s["end"], 2),
                "text": s["text"].strip()
            })

        full_text = "\n".join(
            f"[{o['speaker']}] ({o['start']}s - {o['end']}s): {o['text']}" for o in out
        )
        speaker_text = "\n".join(
            f"[{o['speaker']}]: {o['text']}" for o in out
        )

        if not speaker_text.strip():
            return fail("Transkript bos")

        speakers = list(set(o["speaker"] for o in out))
        elapsed = round(time.time() - t0, 1)

        payload = {
            "success": True,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "language": language,
            "processing_time_seconds": elapsed,
            "file_size_mb": file_size_mb,
            "speakers_found": len(speakers),
            "speakers": speakers,
            "segment_count": len(out),
            "full_text": full_text,
            "speaker_text": speaker_text,
            "diarization_ok": diarization_ok
        }

        webhook(payload)
        logger.info(f"[DONE] {elapsed}s | {len(out)} seg | diar={diarization_ok}")
        return payload

    except Exception as e:
        logger.error(f"[FATAL] {e}", exc_info=True)
        return fail(str(e))

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
