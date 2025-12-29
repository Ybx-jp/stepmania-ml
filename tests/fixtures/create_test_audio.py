import numpy as np
import soundfile as sf

# Create 10 second sine wave for testing
sr = 22050
duration = 10.0
t = np.linspace(0, duration, int(sr * duration))
audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # Quiet sine wave

sf.write('test_audio.wav', audio, sr)
print(f'Created test_audio.wav: {len(audio)} samples, {duration}s')