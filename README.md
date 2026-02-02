# Enhanced Landslide Detection Using Spatial-Channel Attention with ResNet50

## 📋 Project Overview

This project implements an advanced **semantic segmentation model** for detecting landslides in satellite imagery combined with Digital Elevation Model (DEM) data. The model leverages a custom **ResNet50 architecture** enhanced with **Efficient Channel Attention (ECA)** mechanisms to achieve high accuracy in landslide identification across complex topography, particularly in northern Pakistan.

## 🎯 Key Features

- **Multimodal Data Fusion**: Combines RGB satellite imagery with DEM data for richer feature representation
- **ECA Attention Mechanism**: Efficient channel attention with minimal computational overhead (4 parameters, 45.31K FLOPs)
- **Custom ResNet50**: Modified ResNet50 with integrated ECA attention in bottleneck blocks
- **High Accuracy**: Achieves 91%+ accuracy on validation data
- **Interactive Web Interface**: Gradio-based UI for real-time predictions and visualizations
- **GPU Acceleration**: Multi-GPU support via `nn.DataParallel`

---

## 📊 Dataset

### Bijie Landslide Dataset
- **Source**: Kaggle (hanstankman/bijie-landslidedataset)
- **Size**: ~502 MB
- **Structure**:
  ```
  Bijie-landslide-dataset/
  ├── landslide/
  │   ├── image/          (RGB satellite images)
  │   ├── dem/            (Digital Elevation Models)
  │   ├── mask/           (Ground truth segmentation masks)
  │   └── polygon_coordinate/  (Coordinate files)
  └── non-landslide/
      ├── image/
      └── dem/
  ```
- **Data Split**: 67% training / 33% validation
- **Image Size**: Resized to 224×224 pixels
- **Channels**: 4 (RGB + DEM)

---

## 🏗️ Architecture

### Model Components

#### 1. **ECA Attention Module** (`ECAAttention`)
```
Input → GlobalAvgPool → Conv1D → Sigmoid → Channel Weighting → Output
```
- **Purpose**: Adaptively weights feature channels
- **Parameters**: 4
- **FLOPs**: 45.31K (per 512×7×7 feature map)
- **Initialization**: He initialization for stability

#### 2. **Bottleneck Block with ECA**
```
Input → [1×1 Conv] → [3×3 Conv] → [1×1 Conv] 
         ↓ (Skip connection)
         → [ECA Attention] → ReLU → Output
```
- Features: Batch normalization, ReLU activation, shortcut connections

#### 3. **ResNet50 Backbone**
| Layer | Blocks | Output Channels | Stride |
|-------|--------|-----------------|--------|
| Conv1 | 1      | 64              | 2      |
| Conv2 | 3      | 64→256          | 1      |
| Conv3 | 4      | 128→512         | 2      |
| Conv4 | 6      | 256→1024        | 2      |
| Conv5 | 3      | 512→2048        | 2      |

#### 4. **Custom Output Module** (`CustomModel`)
- **UpsampleAndReduceChannels**: 
  - Bilinear upsampling: 7×7 → 224×224
  - Channel reduction: 2048 → 2 channels
- **Output**: 2-channel segmentation map (landslide vs. non-landslide)

### Input Processing
- **Image**: RGB (3 channels) → Normalize with ImageNet statistics
- **DEM**: Grayscale (1 channel) → Normalize to [-1, 1]
- **Concatenation**: 3 + 1 = **4-channel input**
- **Data Augmentation**: Random horizontal flips (50% probability)

---

## 🔧 Technical Stack

| Component | Version/Library |
|-----------|-----------------|
| Deep Learning | PyTorch |
| Computer Vision | torchvision, OpenCV, PIL |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Gradio |
| ML Utilities | scikit-learn (train_test_split) |
| Model Analysis | torchsummary, thop |
| Environment | Google Colab (TPU/GPU) |

---

## 📈 Training Configuration

```python
# Loss Function
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = Adam(lr=0.0001)

# Training Parameters
├── Batch Size: 8
├── Epochs: 20
├── Learning Rate: 1e-4
├── Validation Split: 33%
├── Device: GPU (CUDA) if available
└── Data Workers: 4 (parallel loading)
```

### Training Metrics
| Metric | Training | Validation |
|--------|----------|-----------|
| Initial Loss | 0.1665 | 0.1891 |
| Initial Accuracy | 91.80% | 91.02% |
| Epoch 2 Loss | 0.1783 | 0.1985 |
| Epoch 2 Accuracy | 92.16% | 91.82% |

---

## 📦 Installation & Setup

### Prerequisites
```bash
python >= 3.8
CUDA >= 11.0 (for GPU support)
```

### Install Dependencies
```bash
pip install torch torchvision
pip install kaggle kagglehub
pip install numpy opencv-python pillow
pip install matplotlib scikit-learn tqdm
pip install torchsummary thop
pip install gradio
```

### Setup Kaggle API (for dataset download)
```bash
# Download kaggle.json from https://www.kaggle.com/settings/account
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 🚀 Usage

### 1. Data Preparation
```python
import kagglehub

# Download dataset
path = kagglehub.dataset_download("hanstankman/bijie-landslidedataset")
print("Path to dataset files:", path)
```

### 2. Dataset Loading
```python
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Split data
train_indices, val_indices = train_test_split(
    range(len(dataset)), 
    test_size=0.33, 
    random_state=42
)

# Create loaders
train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)
```

### 3. Model Training
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(epoch, model, train_loader, device)
    val_loss, val_acc = validate(epoch, model, val_loader, device)
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
```

### 4. Model Inference
```python
model.eval()
with torch.no_grad():
    outputs = model(input_tensor)
    predictions = torch.argmax(outputs, dim=1)
```

### 5. Launch Web Interface
```python
import gradio as gr

iface = gr.Interface(
    fn=visualize_segmentation,
    inputs=gr.Image(type="pil"),
    outputs="image",
    live=True
)
iface.launch()
```

---

## 📊 Results & Performance

### Model Metrics
- **Training Accuracy**: ~92%+
- **Validation Accuracy**: ~91%+
- **Model Parameters**: ~23.5M (ResNet50 + ECA)
- **Inference Time**: ~0.2s per image (GPU)

### Visualizations
The model generates 3-subplot comparisons:
1. **Input Image**: RGB satellite image with DEM overlay
2. **Ground Truth Mask**: Actual landslide regions
3. **Predicted Mask**: Model predictions

---

## 🔄 Data Enhancement Functions

### Image Preprocessing
```python
def resize_and_normalize(image, target_size=(224, 224)):
    """Resize and normalize image to [0, 255]"""
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return (image - image.min()) / (image.max() - image.min()) * 255

def histogram_equalization(image):
    """Enhance contrast using YUV color space"""
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def add_noise(image, variance=0.1):
    """Add Gaussian noise for robustness"""
    gauss = np.random.normal(0, variance**0.5, image.shape)
    noisy = image + gauss * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)
```

---

## 📁 Project Structure

```
Enhanced_Landslide_Detection/
├── Enhanced_Landslide Detection.ipynb
├── README.md
├── requirements.txt
└── outputs/
    ├── trained_model.pth
    ├── loss_curves.png
    └── prediction_samples/
```

---

## 🎓 Key Concepts

### Semantic Segmentation
- **Task**: Pixel-level classification (landslide vs. non-landslide)
- **Output**: 2-channel probability map
- **Loss Function**: Cross-Entropy Loss for multi-class classification

### Efficient Channel Attention (ECA)
- **Advantage**: Captures channel interdependencies with minimal overhead
- **Mechanism**: 1D convolution on channel statistics
- **Application**: Applied in every bottleneck block

### Multimodal Fusion
- **RGB Data**: Spectral information from satellite imagery
- **DEM Data**: Topographic/elevation information
- **Fusion Strategy**: Channel concatenation (early fusion)

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA Out of Memory | Reduce batch size from 8 to 4 or 2 |
| Dataset not found | Verify Kaggle API credentials in `~/.kaggle/kaggle.json` |
| Gradio connection error | Add `share=True` in `iface.launch(share=True)` |
| Slow data loading | Reduce `num_workers` or increase to 8 |
| Low accuracy | Increase epochs, use learning rate scheduler |

---

## 📚 References

- **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (2015)
- **ECA-Net**: Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks" (2020)
- **Segmentation**: Long et al., "Fully Convolutional Networks for Semantic Segmentation" (2015)
- **Dataset**: [Bijie Landslide Dataset on Kaggle](https://www.kaggle.com/datasets/hanstankman/bijie-landslidedataset)

---

## 📝 Citation

If you use this project, please cite:
```bibtex
@project{landslide_detection_2024,
  title={Enhanced Landslide Detection Using Spatial-Channel Attention with ResNet50},
  author={Maqsood},
  year={2024},
  note={Kaggle Dataset: hanstankman/bijie-landslidedataset}
}
```

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 👤 Author

**Maqsood**  
Specialization: Deep Learning, Computer Vision, Geospatial Analysis

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

---

## 📞 Support

For issues, questions, or suggestions:
- Create an issue on GitHub
- Check existing documentation
- Refer to Kaggle dataset discussions

---

**Last Updated**: February 2, 2026  
**Status**: Active Development
