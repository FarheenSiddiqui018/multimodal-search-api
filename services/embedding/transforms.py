import io
from PIL import Image
import torch
import torchaudio
from torchvision import transforms

# ––– Image preprocessing (unchanged) ––––––––––––––––––––––
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ––– Audio → Mel-spectrogram → “image” pipeline –––––––––
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=128
)
amp_to_db = torchaudio.transforms.AmplitudeToDB()

audio_image_transform = transforms.Compose([
    # raw_bytes → waveform tensor [time]
    transforms.Lambda(lambda raw: 
        torchaudio.load(io.BytesIO(raw))[0].mean(dim=0)
    ),
    # [time] → [1, time]
    transforms.Lambda(lambda wav: wav.unsqueeze(0)),
    # [1, time] → [1, n_mels, T]
    transforms.Lambda(lambda wav: mel_spec(wav)),
    # convert to dB
    transforms.Lambda(lambda spec: amp_to_db(spec)),
    # [1, n_mels, T] → [3, n_mels, T] so ToPILImage makes an RGB image
    transforms.Lambda(lambda spec: spec.repeat(3,1,1)),
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
