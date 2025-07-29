import os
import numpy as np
import glob
import librosa
import soundfile as sf

def split_path(path):
    split_path = path.split('\\')
    label = split_path[1]
    filename = split_path[2].split('.')[0]
    
    return label, filename

def load_and_concatenate(files):
    music = []
    not_music = []
    global_sample_rate = None
    
    for file in files:
        label, filename = split_path(file)
        waveform, sample_rate = librosa.load(file)
        
        if global_sample_rate is None:
            global_sample_rate = sample_rate
        else:
            assert global_sample_rate == sample_rate
            
        if label == 'Music':
            music.extend(waveform)
        else:
            not_music.extend(waveform)
            
    # Save music and not_music to files
    output_folder = 'concatenated_audio'
    os.makedirs(output_folder, exist_ok=True)
    
    np.save(os.path.join(output_folder, 'music.npy'), np.array(music))
    np.save(os.path.join(output_folder, 'not_music.npy'), np.array(not_music))
    
    return global_sample_rate

def split_into_chunks(audio_data_npy_path, sr, chunk_duration=10):
    audio_data = np.load(audio_data_npy_path)  # Load the audio data from the numpy file
    label = os.path.splitext(os.path.basename(audio_data_npy_path))[0]
    chunk_samples = int(chunk_duration * sr)  # Calculate the number of samples for each 10-second chunk
    chunks = []
    
    # Split audio into chunks of specified duration
    for i in range(0, len(audio_data), chunk_samples):
        chunk = audio_data[i:i + chunk_samples]
        if len(chunk) == chunk_samples:  # Only keep chunks that are exactly 10 seconds
            chunks.append(chunk)
    
    # Create a new folder to store the chunks
    output_folder = os.path.join('10_second_chunks', f'{label}_audio_chunks')
    os.makedirs(output_folder, exist_ok=True)

    # Save each chunk as a separate file
    for idx, chunk in enumerate(chunks):
        chunk_filename = os.path.join(output_folder, f'{label}_chunk_{idx}.wav')
        sf.write(chunk_filename, chunk, sr)

# Get all audio paths
audio_paths = glob('RadioLabels/*/*.wav')

# Load and concatenate all audio files
sample_rate = load_and_concatenate(audio_paths)

# Split concatenated audio into 10-second chunks
os.makedirs('10_second_chunks', exist_ok=True)

split_into_chunks('concatenated_audio\\music.npy', sample_rate)
split_into_chunks('concatenated_audio\\not_music.npy', sample_rate)