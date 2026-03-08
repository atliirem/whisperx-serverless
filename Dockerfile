FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    whisperx \
    runpod \
    pyannote.audio \
    requests \
    soundfile \
    pandas \
    librosa

RUN python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='float32')" || true

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN if [ -n "$HF_TOKEN" ]; then \
    python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', token='${HF_TOKEN}')" || true; \
    fi

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
