# RadioAdBlock-ML

A machine learning project for radio content classification using Convolutional Recurrent Neural Networks (CRNN) to distinguish between music, advertisements, and talking segments in radio streams.

## 🎯 Project Overview

This project implements an automated radio ad detection system using deep learning. It processes audio streams and classifies content into three categories:
- **Music** 🎵
- **Advertisements** 📢  
- **Talking/Speech** 🗣️

**🔗 Integration**: This ML model is used by the [RadioAdblock](https://github.com/JoshuaLeeZhang/RadioAdblock) radio player application to automatically detect and skip advertisements in real-time radio streams.

## 🏗️ Architecture

- **Model**: CRNN (Convolutional Recurrent Neural Network)
- **Features**: MFCC (Mel-Frequency Cepstral Coefficients)
- **Audio Processing**: 10-second audio chunks
- **Framework**: PyTorch

## 📁 Project Structure

```
RadioML/
├── CRNN.py                    # CRNN model architecture
├── train_model.py             # Model training script
├── feature_engineering.py     # Audio feature extraction
├── create_audio_chunks.py     # Audio preprocessing
├── NpyDataset.py             # PyTorch dataset for .npy files
├── main.py                   # Main execution script
├── exportsd.py               # Model export utilities
├── radioML.ipynb             # Jupyter notebook for experimentation
├── RadioLabels/              # Training data
│   ├── Ads/                  # Advertisement samples
│   ├── Music/                # Music samples
│   └── Talking/              # Speech samples
├── 10_second_chunks/         # Processed audio chunks
├── mfcc/                     # MFCC feature files
└── models/                   # Saved model checkpoints
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchaudio librosa numpy scikit-learn matplotlib jupyter
```

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/JoshuaLeeZhang/RadioAdBlock-ML.git
   cd RadioAdBlock-ML
   ```

2. **Prepare your audio data**
   - Place audio files in appropriate `RadioLabels/` subdirectories
   - Supported formats: WAV, MP3, FLAC

3. **Process audio data**
   ```bash
   python create_audio_chunks.py
   python feature_engineering.py
   ```

4. **Train the model**
   ```bash
   python train_model.py
   ```

## 📊 Usage

### Training
```bash
python main.py --mode train --epochs 50 --batch_size 32
```

### Inference
```bash
python main.py --mode predict --audio_file path/to/audio.wav
```

### Jupyter Notebook
For interactive experimentation:
```bash
jupyter notebook radioML.ipynb
```

## 🔧 Model Details

- **Input**: MFCC features from 10-second audio segments
- **Architecture**: Convolutional layers + LSTM/GRU + Dense layers
- **Output**: 3-class probability distribution (Music, Ads, Talking)
- **Training**: Cross-entropy loss with Adam optimizer

## 📈 Performance

The model achieves classification accuracy for radio content detection. Detailed metrics and evaluation results can be found in the training logs and Jupyter notebook.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request