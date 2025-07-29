import glob
import os
import numpy as np

import torchaudio
import torchaudio.transforms as T

# Load the audio chunks

music_audio_chunk_paths = glob('10_second_chunks/music_audio_chunks/*.wav')
not_music_audio_chunk_paths = glob('10_second_chunks/not_music_audio_chunks/*.wav')

# Verify that all audio chunks have the same sample rate of 22050 Hz

SAMPLE_RATE = 22050

def verify_same_sample_rate(audio_paths):
    for audio_path in audio_paths:
        _, sr = torchaudio.load(audio_path)
        assert sr == SAMPLE_RATE
        
def compute_and_store_mfccs(audio_chunk_paths, label):
    os.makedirs(f'mfcc/{label}_mfcc', exist_ok=True)
    
    for path in audio_chunk_paths:
        waveform = torchaudio.load(path)[0]
        mfcc = mfcc_transform(waveform)
        output_path = f'mfcc/{label}_mfcc/mfcc_{os.path.basename(path).split('.')[0]}.npy'        
        np.save(output_path, mfcc.numpy())

# Set up the MFCC Transformer 

mfcc_params = {
    'sample_rate': SAMPLE_RATE,
    'n_mfcc': 20,
    'melkwargs': {
        'n_fft': 2048,
        'n_mels': 128,
        'hop_length': 512,
        'mel_scale': 'htk'
    }
}

mfcc_transform = T.MFCC(**mfcc_params)
        
os.makedirs('mfcc', exist_ok=True)

verify_same_sample_rate(music_audio_chunk_paths)
verify_same_sample_rate(not_music_audio_chunk_paths)

compute_and_store_mfccs(music_audio_chunk_paths, 'music')   
compute_and_store_mfccs(not_music_audio_chunk_paths, 'not_music')