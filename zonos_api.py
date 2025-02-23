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

from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device

# Constants and globals
REFERENCE_AUDIO_PATH = Path("ref_audio/reference-audio.wav")
DEFAULT_MODEL = "Zyphra/Zonos-v0.1-hybrid"
SPEAKER_EMBEDDING = None
SPEAKER_AUDIO_PATH = None

# Initialize FastAPI app
app = FastAPI(title="Zonos TTS API")

# Load model at startup
print(f"Loading {DEFAULT_MODEL} model...")
CURRENT_MODEL = Zonos.from_pretrained(DEFAULT_MODEL, device=device)
CURRENT_MODEL.requires_grad_(False).eval()
print(f"{DEFAULT_MODEL} model loaded successfully!")

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
async def synthesize_speech(request: TTSRequest):
    """Generate speech from text with optional speaker cloning"""
    try:
        if not REFERENCE_AUDIO_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Reference audio file not found at {REFERENCE_AUDIO_PATH}"
            )

        # Handle speaker audio if provided
        global SPEAKER_AUDIO_PATH, SPEAKER_EMBEDDING
        if "speaker" not in request.unconditional_keys:
            # Try to load from cache first
            print("Loading reference audio...")
            wav, sr = torchaudio.load(REFERENCE_AUDIO_PATH)
            print("Making speaker embedding...")
            SPEAKER_EMBEDDING = CURRENT_MODEL.make_speaker_embedding(wav, sr)
            print("creating speaker embedding")
            SPEAKER_EMBEDDING = SPEAKER_EMBEDDING.to(device, dtype=torch.bfloat16)
            SPEAKER_AUDIO_PATH = str(REFERENCE_AUDIO_PATH)

        # Set seed
        print("Setting the seed...")
        if request.randomize_seed:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            seed = request.seed
        torch.manual_seed(seed)

        # Prepare conditioning
        print("Preparing tensor...")
        emotion_tensor = torch.tensor(request.emotion.to_list(), device=device)
        print("Creating vq tensor...")
        vq_tensor = torch.tensor([request.vq_single] * 8, device=device).unsqueeze(0)

        print("Making cond dict...")
        cond_dict = make_cond_dict(
            text=request.text,
            language=request.language,
            speaker=SPEAKER_EMBEDDING,
            emotion=emotion_tensor,
            vqscore_8=vq_tensor,
            fmax=request.fmax,
            pitch_std=request.pitch_std,
            speaking_rate=request.speaking_rate,
            dnsmos_ovrl=request.dnsmos_ovrl,
            speaker_noised=request.speaker_noised,
            device=device,
            unconditional_keys=request.unconditional_keys,
        )
        print("Preparing conditioning...")
        conditioning = CURRENT_MODEL.prepare_conditioning(cond_dict)

        # Generate audio
        print("Generating audio...")
        codes = CURRENT_MODEL.generate(
            prefix_conditioning=conditioning,
            max_new_tokens=86 * 30,
            cfg_scale=request.cfg_scale,
            batch_size=1
        )

        # Convert to audio
        print("Converting to audio...")
        wav_out = CURRENT_MODEL.autoencoder.decode(codes).cpu().detach()    
        sr_out = CURRENT_MODEL.autoencoder.sampling_rate
        if wav_out.dim() == 3 and wav_out.size(0) == 1:
            wav_out = wav_out.squeeze(0)  # -> shape: (channels, time)

        # Keep only the first channel (optional) if you want mono
        if wav_out.size(0) > 1:
            wav_out = wav_out[0:1, :]

        # Convert tensor to bytes
        buffer = BytesIO()
        torchaudio.save(buffer, wav_out, sr_out, format="wav")
        buffer.seek(0)

        return StreamingResponse(
            buffer, 
            media_type="audio/wav",
            headers={
                'Content-Disposition': 'attachment; filename="generated_speech.wav"'
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("zonos_api:app", host="0.0.0.0", port=8000, reload=True)