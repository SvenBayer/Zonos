import torch
import torchaudio
from pathlib import Path
import json

from zonos.model import Zonos
from zonos.utils import DEFAULT_DEVICE as device

def cache_synthesis_params():
    """Pre-compute and cache synthesis parameters"""
    CACHE_DIR = Path("cache")
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Load model temporarily
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    
    # Create speaker embedding
    wav, sr = torchaudio.load("ref_audio/reference-audio.wav")
    speaker_embedding = model.make_speaker_embedding(wav.to(device), sr)
    torch.save(speaker_embedding, CACHE_DIR / "speaker_embedding.pt")
    
    # Create emotion tensor
    emotion_values = [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2]
    emotion_tensor = torch.tensor(emotion_values, device=device)
    torch.save(emotion_tensor, CACHE_DIR / "emotion_tensor.pt")
    
    # Create VQ tensor
    vq_tensor = torch.tensor([[0.78] * 8], device=device)
    torch.save(vq_tensor, CACHE_DIR / "vq_tensor.pt")
    
    # Save synthesis parameters
    synthesis_params = {
        "language": "en-us",
        "fmax": 48000,
        "pitch_std": 45.0,
        "speaking_rate": 15.0,
        "dnsmos_ovrl": 4.0,
        "speaker_noised": False,
        "device": str(device),  # Convert device to string
        "unconditional_keys": ["pitch_std"]
    }
    
    with open(CACHE_DIR / "synthesis_cache.json", 'w') as f:
        json.dump({"synthesis_params": synthesis_params}, f, indent=2)

if __name__ == "__main__":
    cache_synthesis_params()