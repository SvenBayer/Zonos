from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from typing import List
import torch
import torchaudio
from pathlib import Path
import tempfile
import uvicorn
import base64
from io import BytesIO
import json

from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device

# Constants and globals
REFERENCE_AUDIO_PATH = Path("ref_audio/reference-audio.wav")
DEFAULT_MODEL = "Zyphra/Zonos-v0.1-transformer"
SPEAKER_EMBEDDING = None
CURRENT_MODEL = None
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_CONFIG = CACHE_DIR / "synthesis_cache.json"
CACHED_PARAMS = None

# Initialize FastAPI app
app = FastAPI(title="Zonos TTS API")

def load_cached_params():
    """Load pre-computed parameters from cache"""
    with open(CACHE_CONFIG, 'r') as f:
        config = json.load(f)
    
    cached = {
        "emotion_tensor": torch.load(CACHE_DIR / "emotion_tensor.pt").to(device),
        "vq_tensor": torch.load(CACHE_DIR / "vq_tensor.pt").to(device),
        "speaker_embedding": torch.load(CACHE_DIR / "speaker_embedding.pt").to(device),
        "synthesis_params": config["synthesis_params"]
    }
    return cached

@app.on_event("startup")
async def startup_event():
    """Initialize model and speaker embedding on startup"""
    global CURRENT_MODEL, SPEAKER_EMBEDDING, CACHED_PARAMS
    # Load model and create speaker embedding at startup
    print(f"Loading {DEFAULT_MODEL} model...")
    CURRENT_MODEL = Zonos.from_pretrained(DEFAULT_MODEL, device=device)
    print("Model requires grad eval...")
    CURRENT_MODEL.requires_grad_(False).eval()
    print(f"{DEFAULT_MODEL} model loaded successfully!")

    # Process reference audio at startup
    print("Processing reference audio...")
    if REFERENCE_AUDIO_PATH.exists():
        print("Loading torch audio...")
        wav, sr = torchaudio.load(REFERENCE_AUDIO_PATH)
        print("Wav to device moving to GPU...")
        wav = wav.to(device)  # Move to GPU once
        print("Make speaker embedding....")
        SPEAKER_EMBEDDING = CURRENT_MODEL.make_speaker_embedding(wav, sr)
        print("Speaker embedding to device...")
        SPEAKER_EMBEDDING = SPEAKER_EMBEDDING.to(device, dtype=torch.bfloat16)
        print("Freeing up memory...")
        del wav  # Free up memory
        del sr
        print("Reference audio processed successfully!")
    else:
        print(f"Warning: Reference audio not found at {REFERENCE_AUDIO_PATH}")

    # Load cached parameters
    CACHED_PARAMS = load_cached_params()
    print("Model and cached parameters loaded successfully!")

class Emotion(BaseModel):
    """Emotion values for speech synthesis"""
    happiness: float = Field(default=1.0, ge=0.0, le=1.0)
    sadness: float = Field(default=0.05, ge=0.0, le=1.0)
    disgust: float = Field(default=0.05, ge=0.0, le=1.0)
    fear: float = Field(default=0.05, ge=0.0, le=1.0)
    surprise: float = Field(default=0.05, ge=0.0, le=1.0)
    anger: float = Field(default=0.05, ge=0.0, le=1.0)
    other: float = Field(default=0.1, ge=0.0, le=1.0)
    neutral: float = Field(default=0.2, ge=0.0, le=1.0)

    def to_list(self) -> List[float]:
        """Convert emotion object to list of values"""
        return [
            self.happiness,
            self.sadness,
            self.disgust,
            self.fear,
            self.surprise,
            self.anger,
            self.other,
            self.neutral
        ]

class TTSRequest(BaseModel):
    """Request model for text-to-speech synthesis"""
    text: str
    language: str = "en-us"
    emotion: Emotion = Field(default_factory=Emotion)
    vq_single: float = 0.78
    fmax: float = 48000
    pitch_std: float = 45.0
    speaking_rate: float = 15.0
    dnsmos_ovrl: float = 4.0
    speaker_noised: bool = False
    cfg_scale: float = 2.0
    seed: int = 420
    randomize_seed: bool = True
    # "speaker", "emotion", "vqscore_8", "fmax", "pitch_std", "speaking_rate", "dnsmos_ovrl", "speaker_noised"
    unconditional_keys: List[str] = ["pitch_std"]

class TextRequest(BaseModel):
    """Simple request with just text"""
    text: str

@app.get("/models")
async def get_supported_models():
    """Get list of supported models"""
    supported_models = []
    if "transformer" in ZonosBackbone.supported_architectures:
        supported_models.append(DEFAULT_MODEL)
    if "hybrid" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-hybrid")
    return {"supported_models": supported_models}

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported language codes"""
    return {"supported_languages": supported_language_codes}

@app.post("/synthesize")
async def synthesize_speech(request: TextRequest):
    """Generate speech from text using cached parameters"""
    try:
        with torch.no_grad():
            # Use cached parameters
            cond_dict = make_cond_dict(
                text=request.text,
                language=CACHED_PARAMS["synthesis_params"]["language"],
                speaker=CACHED_PARAMS["speaker_embedding"],
                emotion=CACHED_PARAMS["emotion_tensor"],
                vqscore_8=CACHED_PARAMS["vq_tensor"],
                **CACHED_PARAMS["synthesis_params"]
            )
            
            conditioning = CURRENT_MODEL.prepare_conditioning(cond_dict)
            codes = CURRENT_MODEL.generate(
                prefix_conditioning=conditioning,
                max_new_tokens=86 * 30,
                cfg_scale=2.0,
                batch_size=1
            )

            wav_out = CURRENT_MODEL.autoencoder.decode(codes).cpu()
            del codes, conditioning
            torch.cuda.empty_cache()

        sr_out = CURRENT_MODEL.autoencoder.sampling_rate
        wav_out = wav_out.squeeze(0) if wav_out.dim() == 3 else wav_out[0:1, :]

        buffer = BytesIO()
        torchaudio.save(buffer, wav_out, sr_out, format="wav")
        buffer.seek(0)

        return StreamingResponse(
            buffer, 
            media_type="audio/wav",
            headers={'Content-Disposition': 'attachment; filename="generated_speech.wav"'}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("zonos_api:app", host="0.0.0.0", port=8000, reload=False)