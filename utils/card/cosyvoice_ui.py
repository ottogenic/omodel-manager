"""Isolated CosyVoice 3 Gradio test UI; voice references stay in a mounted directory."""

import json
import os
import re
import tempfile
import threading
import hashlib
import time

import gradio as gr
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

if os.environ.get("COSYVOICE_ACCEL") == "vllm":
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav

MODEL_DIR = "/model"
VOICE_DIR = "/voices"
os.makedirs(VOICE_DIR, exist_ok=True)
ACCEL = os.environ.get("COSYVOICE_ACCEL", "fp16")
if ACCEL not in ("native", "fp16", "vllm", "trt"):
    raise ValueError("Unknown CosyVoice acceleration setting")
print(f"CosyVoice acceleration: {ACCEL}", flush=True)
model = AutoModel(model_dir=MODEL_DIR, load_trt=ACCEL == "trt",
                  load_vllm=ACCEL == "vllm", fp16=ACCEL == "fp16")
synthesis_lock = threading.Lock()
cached_profiles = {}


def reset_stream_hop():
    """Upstream mutates this shared field to 100 while streaming a request.

    Its initial 25-token hop is required for a low-latency first chunk on
    subsequent requests too. All synthesis is serialized by synthesis_lock.
    """
    model.model.token_hop_len = 25


def saved_names():
    return sorted(name[:-5] for name in os.listdir(VOICE_DIR)
                  if name.endswith(".json") and re.fullmatch(r"[a-zA-Z0-9_-]+", name[:-5])
                  and os.path.isfile(os.path.join(VOICE_DIR, name[:-5] + ".wav")))


def saved_reference(name):
    if name not in saved_names():
        raise ValueError("Unknown saved voice")
    with open(os.path.join(VOICE_DIR, name + ".json"), encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not metadata.get("consent") or not metadata.get("transcript"):
        raise ValueError("Saved voice needs consent and an exact transcript")
    return os.path.join(VOICE_DIR, name + ".wav"), metadata["transcript"]


def cached_reference(name, wav, transcript):
    """Prepare the upstream zero-shot speaker once per unchanged reference.

    Called under synthesis_lock, shared with both the test UI and app API.
    The source WAV and transcript on disk remain authoritative across restarts.
    """
    stat = os.stat(wav)
    fingerprint = (stat.st_mtime_ns, stat.st_size, transcript)
    previous = cached_profiles.get(name)
    if previous and previous[0] == fingerprint:
        return previous[1]
    speaker_id = 'saved_' + hashlib.sha256(repr((name, fingerprint)).encode()).hexdigest()[:24]
    t0 = time.monotonic()
    model.add_zero_shot_spk("You are a helpful assistant.<|endofprompt|>" + transcript, wav, speaker_id)
    cached_profiles[name] = (fingerprint, speaker_id)
    if previous:
        model.frontend.spk2info.pop(previous[1], None)
    print(f"[perf] prepared saved voice {name} in {time.monotonic() - t0:.3f}s", flush=True)
    return speaker_id


def save_voice(name, audio, transcript, consent):
    name = (name or "").strip()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,50}", name):
        raise gr.Error("Choose a name using 1–50 letters, numbers, _ or -.")
    if not audio or not transcript.strip() or not consent:
        raise gr.Error("A reference recording, exact transcript and consent are required.")
    destination = os.path.join(VOICE_DIR, name + ".wav")
    metadata = os.path.join(VOICE_DIR, name + ".json")
    if os.path.exists(destination) or os.path.exists(metadata):
        raise gr.Error("Voice name already exists; choose another name.")
    # Normalize to a private, persistent WAV; never retain Gradio's temporary upload.
    wav = load_wav(audio, target_sr=16000, min_sr=16000)
    if not 1 <= wav.shape[1] / 16000 <= 30:
        raise gr.Error("Reference audio must be 1–30 seconds long.")
    fd, temporary = tempfile.mkstemp(dir=VOICE_DIR, suffix=".wav")
    os.close(fd)
    try:
        torchaudio.save(temporary, wav, 16000)
        os.replace(temporary, destination)
        with open(metadata, "x", encoding="utf-8") as stream:
            json.dump({"transcript": transcript.strip(), "consent": True}, stream)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return f"Saved {name} (reference WAV + transcript).", gr.update(choices=saved_names(), value=name)


def generate(text, mode, voice_name, audio, transcript, instruction):
    text = (text or "").strip()
    if not text or len(text) > 500:
        raise gr.Error("Enter 1–500 characters of speech text.")
    if voice_name:
        try:
            audio, transcript = saved_reference(voice_name)
        except ValueError as error:
            raise gr.Error(str(error)) from error
    if not audio:
        raise gr.Error("Upload reference audio or select a saved voice.")
    reference = load_wav(audio, target_sr=16000, min_sr=16000)
    if not 1 <= reference.shape[1] / 16000 <= 30:
        raise gr.Error("Reference audio must be 1–30 seconds long.")
    if mode == "Clone":
        if not (transcript or "").strip():
            raise gr.Error("Cloning requires the exact reference transcript.")
        # Uploaded audio uses the normal path; saved references use upstream's
        # cached zero-shot speaker profile shared with otalk-app.
        chunks = None if voice_name else model.inference_zero_shot(text,
            "You are a helpful assistant.<|endofprompt|>" + transcript.strip(), audio, stream=True)
    else:
        if not (instruction or "").strip():
            raise gr.Error("Enter a style instruction.")
        directive = instruction.strip()
        if "<|endofprompt|>" not in directive:
            directive = "You are a helpful assistant. " + directive + "<|endofprompt|>"
        chunks = model.inference_instruct2(text, directive, audio, stream=True)
    with synthesis_lock:
        if mode == "Clone" and voice_name:
            speaker_id = cached_reference(voice_name, audio, transcript)
            chunks = model.inference_zero_shot(text, "", audio,
                zero_shot_spk_id=speaker_id, stream=True)
        reset_stream_hop()
        for chunk in chunks:
            samples = chunk["tts_speech"].squeeze().detach().cpu().numpy()
            if samples.size:
                yield (model.sample_rate, samples)


with gr.Blocks(title="CosyVoice 3 — RTX 4070 test") as demo:
    gr.Markdown("# CosyVoice 3 — experimental test\nReference audio stays on this machine. Test `[laughter]`, `<strong>emphasis</strong>`, and style instructions; judge the audio by listening. No voice is saved unless you click Save voice.")
    text = gr.Textbox(label="Speech text", value="Wait—[laughter] that is <strong>really</strong> surprising.")
    mode = gr.Radio(["Clone", "Style instruction"], value="Clone", label="Mode")
    voice = gr.Dropdown(saved_names(), label="Saved reference voice (optional)")
    audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference voice recording")
    transcript = gr.Textbox(label="Exact transcript of reference recording (Clone mode)")
    instruction = gr.Textbox(label="Style instruction (Style instruction mode)", placeholder="Speak slowly, sounding frightened.")
    output = gr.Audio(label="Synthesized speech", streaming=True, autoplay=True)
    gr.Button("Generate").click(generate, [text, mode, voice, audio, transcript, instruction], output)
    gr.Markdown("## Save a reference voice for later tests")
    name = gr.Textbox(label="Voice name")
    consent = gr.Checkbox(label="I have permission to use this voice recording")
    message = gr.Textbox(label="Save status", interactive=False)
    gr.Button("Save voice").click(save_voice, [name, audio, transcript, consent], [message, voice])

# Warm saved voices before accepting requests, so the first user turn does not
# pay for reference extraction. A newly saved voice is prepared on first use.
with synthesis_lock:
    for name in saved_names():
        try:
            wav, transcript = saved_reference(name)
            cached_reference(name, wav, transcript)
        except (OSError, ValueError) as error:
            print(f"Could not prepare saved voice {name}: {error}", flush=True)

if ACCEL == "vllm":
    # vLLM's newer Pydantic breaks this pinned Gradio release's API schema.
    # Keep the separate benchmark API available without altering native UI.
    app = FastAPI()

    @app.get("/")
    def benchmark_landing():
        return HTMLResponse("<h1>CosyVoice 3 native vLLM benchmark — test UI unavailable</h1>")
else:
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=8000,
        share=False, show_error=True, prevent_thread_lock=True)
    app = demo.app


@app.get("/api/voices")
def voice_list():
    return {"voices": saved_names()}


@app.post("/api/speech")
async def speech(request: Request):
    if int(request.headers.get("content-length", "0")) > 10_000:
        raise HTTPException(413, "Request too large")
    try:
        payload = await request.json()
    except ValueError as error:
        raise HTTPException(400, "Invalid JSON") from error
    if not isinstance(payload, dict):
        raise HTTPException(400, "Invalid speech request")
    text, name = payload.get("text"), payload.get("voice")
    if not isinstance(text, str) or not text.strip() or len(text) > 3500 or not isinstance(name, str):
        raise HTTPException(400, "Invalid speech request")
    try:
        wav, transcript = saved_reference(name)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error

    def pcm_chunks():
        with synthesis_lock:
            speaker_id = cached_reference(name, wav, transcript)
            reset_stream_hop()
            for chunk in model.inference_zero_shot(text.strip(), "", wav,
                    zero_shot_spk_id=speaker_id, stream=True):
                samples = chunk["tts_speech"].squeeze().detach().cpu().clamp(-1, 1)
                if samples.numel():
                    yield (samples * 32767).to(dtype=torch.int16).numpy().tobytes()

    return StreamingResponse(pcm_chunks(), media_type="audio/pcm",
        headers={"Cache-Control": "no-store", "X-Sample-Rate": str(model.sample_rate)})


if ACCEL == "vllm":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    threading.Event().wait()
