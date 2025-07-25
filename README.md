# Deep Convolutional Bidirectional LSTM for Epileptic Seizure Detection

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository implements a deep learning approach for automatic epileptic seizure detection from raw EEG signals using a hybrid **Convolutional Neural Network (CNN) + Bidirectional Long Short-Term Memory (Bi-LSTM)** architecture. The system achieves **100% accuracy** on binary classification tasks without requiring manual feature extraction.

## Research Background

This implementation is based on the research paper:
> **"Deep Convolutional Bidirectional LSTM Recurrent Neural Network for Epileptic Seizure Detection"**  
> *Ahmed M. Abdelhameed, Hisham G. Daoud, Magdy Bayoumi*  
> University of Louisiana at Lafayette

### Key Features

- 🧠 **End-to-end learning**: Works directly on raw EEG signals without manual feature extraction
- 🎯 **High accuracy**: Achieves 100% accuracy on normal vs. seizure classification
- 🔄 **Robust validation**: Uses 10-fold cross-validation for reliable performance evaluation
- ⚡ **Efficient architecture**: CNN front-end reduces sequence length for faster LSTM processing
- 📊 **Multiple classification tasks**: Supports binary and multi-class EEG classification

## Architecture

The model combines two powerful deep learning components:

1. **1D CNN Frontend**: 
   - Automatically extracts features from raw EEG signals
   - Reduces sequence length for efficient processing
   - Uses batch normalization and ReLU activation

2. **Bidirectional LSTM Backend**:
   - Captures temporal dependencies in both directions
   - Preserves long-term patterns in EEG sequences
   - Uses dropout for regularization

```
Raw EEG Signal → CNN Layers → Bi-LSTM → Dense → Classification
   (4097 pts)    (Feature     (Temporal   (Output)
                 Extraction)   Modeling)
```

## Dataset

The implementation uses the publicly available **Bonn University EEG dataset**:

- **Source**: Department of Epileptology, University of Bonn
- **Size**: 500 EEG segments (100 per class)
- **Duration**: 23.6 seconds per segment
- **Sampling Rate**: 173.61 Hz
- **Length**: 4097 data points per segment
- **Classes**:
  - **Set A**: Normal subjects, eyes open
  - **Set B**: Normal subjects, eyes closed  
  - **Set C**: Interictal (seizure-free) from epileptic patients
  - **Set D**: Interictal from epileptogenic zone
  - **Set E**: Ictal (during seizure) recordings

## Results

### Binary Classification (Normal vs. Seizure)
| Task | Accuracy | Sensitivity | Specificity |
|------|----------|-------------|-------------|
| A vs E | **100%** | 100% | 100% |
| B vs E | **99%** | 100% | 98% |

### Multi-class Classification (Normal vs. Interictal vs. Ictal)
| Task | Accuracy | Sensitivity | Specificity |
|------|----------|-------------|-------------|
| A/D/E | **98.66%** | 98% | 99% |
| B/D/E | **98.89%** | 98.33% | 99.16% |

## Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (recommended)

### Dependencies
```bash
pip install tensorflow>=2.8.0
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install scipy
```

### Clone Repository
```bash
git clone https://github.com/yourusername/eeg-seizure-detection.git
cd eeg-seizure-detection
```

## Usage

### 1. Data Preparation
Download the Bonn University EEG dataset and organize it as follows:
```
data3/
├── Z001.txt  # Normal, eyes open
├── Z002.txt
├── ...
├── S001.txt  # Seizure recordings
├── S002.txt
└── ...
```

### 2. Training the Model
```python
# Open the Jupyter notebook
jupyter notebook Newcas-Tensorflow-100-Copy1.ipynb

# Or run the training script directly
python train_model.py
```

### 3. Model Configuration
Key hyperparameters in the notebook:
```python
# Data parameters
NUMBER_OF_SAMPLES = 200
ORIGINAL_SEQUENCE_SIZE = 4097

# Training parameters
NUM_EPOCHS = 20
BATCH_SIZE = 16
N_SPLITS = 10  # for cross-validation

# Model architecture
CONV_FILTERS = 16
CONV_KERNEL_SIZE = 5
LSTM_UNITS = 20
LSTM_DROPOUT = 0.1
```

## Model Architecture Details

### CNN Layers
- **4 Convolutional Layers**: 1D convolutions with 16 filters
- **Kernel Size**: 5
- **Activation**: ReLU
- **Normalization**: Batch normalization after each conv layer
- **Pooling**: Max pooling with size 3

### Bi-LSTM Layer
- **Units**: 20 LSTM cells per direction (40 total)
- **Dropout**: 0.1 for regularization
- **Output**: Takes final timestep output for classification

### Output Layer
- **Dense Layer**: 1 neuron with sigmoid activation
- **Loss Function**: Binary crossentropy
- **Optimizer**: RMSprop

## File Structure

```
├── Newcas-Tensorflow-100-Copy1.ipynb  # Main implementation notebook
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data3/                            # EEG dataset directory
├── models/                           # Saved model weights
├── results/                          # Training results and plots
└── utils/                           # Helper functions
    ├── data_loader.py               # Data loading utilities
    ├── model_utils.py               # Model architecture helpers
    └── evaluation.py               # Metrics and visualization
```

## Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall classification correctness
- **Sensitivity**: True positive rate (seizure detection rate)
- **Specificity**: True negative rate (normal classification rate)
- **Confusion Matrix**: Detailed classification breakdown

## Cross-Validation

The implementation uses **10-fold cross-validation** to ensure robust performance:
- Data split: 80% training, 20% validation per fold
- Each fold tested independently
- Results averaged across all folds
- Standard deviation reported for reliability

## Reproducibility

To ensure reproducible results:
- Fixed random seeds for all libraries
- Deterministic operations enabled
- Cross-validation with stratified splits
- Consistent data preprocessing

## Comparison with State-of-the-Art

| Method | Approach | Accuracy | Dataset Split |
|--------|----------|----------|---------------|
| **Our Method** | **CNN + Bi-LSTM** | **98.89%** | **10-fold CV** |
| Acharya et al. [12] | Deep CNN | 88.7% | 10-fold CV |
| Güler et al. [9] | RNN + Lyapunov | 96.79% | 80/20 split |
| Orhan et al. [10] | k-means + MLP | 96.67% | N/A |

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{abdelhameed2018deep,
  title={Deep Convolutional Bidirectional LSTM Recurrent Neural Network for Epileptic Seizure Detection},
  author={Abdelhameed, Ahmed M and Daoud, Hisham G and Bayoumi, Magdy},
  booktitle={2018 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={139--143},
  year={2018},
  organization={IEEE}
}
```

## Future Work

- [ ] Real-time seizure detection implementation
- [ ] Extension to multi-channel EEG data
- [ ] Patient-specific model adaptation
- [ ] Mobile/edge deployment optimization
- [ ] Integration with clinical decision support systems

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Bonn University for providing the public EEG dataset
- University of Louisiana at Lafayette research team
- TensorFlow and Keras communities for excellent documentation

## Contact

For questions and support:
- **Email**: [ahmed_barbary@yahoo.com] [abdelhameed.ahmed@mayo.edu]
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/eeg-seizure-detection/issues)

---

⭐ **Star this repository if you find it helpful!** ⭐
