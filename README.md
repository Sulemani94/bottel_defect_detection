🔬 Medical Bottle Defect Detection using YOLOv8
> Real-time detection of **cracks** and **particles** in medical glass bottles using a custom-trained YOLOv8 object detection model.
---
📌 Project Overview
In pharmaceutical and medical manufacturing, bottle integrity is critical. Even minor defects like micro-cracks or foreign particle contamination can compromise product safety. This project implements an automated visual inspection system using deep learning to detect such defects in grayscale bottle images.
The model was trained on a custom dataset of medical bottles and is capable of detecting:
Defect Class	Description
`Crack`	Structural cracks or fractures on the bottle surface
`Particle`	Foreign particle contamination inside the bottle
---
🖼️ Sample Predictions
The model successfully detects both defect types with reasonable confidence on unseen bottle images:
Crack detected at ~75–83% confidence (purple bounding box)
Particle detected at ~75–80% confidence (green bounding box)
> Predictions are shown on real grayscale medical bottle images under controlled lighting.
---
📊 Training Results
The model was trained for ~220 epochs using the YOLOv8 architecture.
Loss Curves
Metric	Trend
`train/box_loss`	Consistently decreasing ✅
`train/cls_loss`	Steeply decreasing, converged ✅
`train/dfl_loss`	Steadily decreasing ✅
`val/box_loss`	Stable with minor fluctuations ✅
`val/cls_loss`	Converged well ✅
Performance Metrics (at convergence)
Metric	Approximate Value
Precision (B)	~0.60–0.65
Recall (B)	~0.65–0.70
mAP@50	~0.60–0.65
> Note: High variance in validation metrics is expected given the small and visually challenging dataset (subtle cracks under varying lighting conditions).
---
🏗️ Model Architecture
Base Model: YOLOv8 (Ultralytics)
Task: Object Detection
Input: Grayscale images of medical bottles
Classes: 2 (`crack`, `particle`)
Training Epochs: ~220
Framework: PyTorch + Ultralytics
---
📁 Project Structure
```
medical-bottle-defect-detection/
│
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── runs/
│   └── detect/
│       └── train/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           └── results.png
│
├── predict_samples/
│   ├── pc1.PNG
│   ├── pc2.PNG
│   └── pc3.PNG
│
├── data.yaml
├── train.py
├── predict.py
└── README.md
```
---
🚀 Getting Started
1. Clone the Repository
```bash
git clone https://github.com/your-username/medical-bottle-defect-detection.git
cd medical-bottle-defect-detection
```
2. Install Dependencies
```bash
pip install ultralytics opencv-python matplotlib
```
3. Train the Model
```bash
python train.py
```
Or directly with Ultralytics CLI:
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=220 imgsz=640
```
4. Run Inference
```bash
python predict.py --source your_image.jpg --weights runs/detect/train/weights/best.pt
```
---
⚙️ `data.yaml` Configuration
```yaml
path: ./data
train: images/train
val: images/val

nc: 2
names: ['crack', 'particle']
```
---
📈 Training Script (`train.py`)
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load pretrained YOLOv8 nano

model.train(
    data='data.yaml',
    epochs=220,
    imgsz=640,
    batch=16,
    name='bottle_defect_detector'
)
```
---
🔍 Inference Script (`predict.py`)
```python
from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train/weights/best.pt')

results = model.predict(source='predict_samples/', save=True, conf=0.4)
```
---
🧠 Key Observations & Challenges
Subtle defects: Cracks are very thin and low-contrast on grayscale images, making detection inherently difficult.
Small particles: Foreign particles are tiny, contributing to lower mAP@50-95 scores.
Validation variance: High epoch-to-epoch fluctuation in val metrics is typical for small medical datasets.
Improvement potential: Data augmentation, larger dataset, and higher-resolution input could significantly improve recall.
---
🔮 Future Work
[ ] Expand dataset with more diverse bottle types and defect severities
[ ] Apply data augmentation (brightness, contrast, rotation) to reduce overfitting
[ ] Experiment with YOLOv8m/l for higher accuracy
[ ] Integrate into a real-time industrial inspection pipeline
[ ] Add defect severity scoring (minor / major / critical)
---
🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
---
📄 License
This project is licensed under the MIT License — see the LICENSE file for details.
---
👤 Author
Khizar Hayat 
Computer Vision & Deep Learning Enthusiast  
LinkedIn | GitHub
---
> ⭐ If you found this project helpful, please give it a star!
