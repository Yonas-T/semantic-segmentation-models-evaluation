# 🏙️ Cityscapes Semantic Segmentation Benchmark

An end-to-end benchmarking platform that trains, evaluates, and contrasts **SegFormer-B2** (Transformer) against **DeepLabV3+** (CNN) on the Cityscapes urban dataset — with an interactive Next.js web portal for real-time inference and analytics.

![Dashboard Preview](docs/dashboard-preview.png)

---

## 📂 Project Structure

```
cityscapes-benchmark/
├── ml-pipeline/                        # Python Machine Learning Code
│   ├── src/
│   │   ├── dataset.py                  # PyTorch Dataset & Augmentations (Cityscapes 19-class)
│   │   ├── models.py                   # SegFormer-B2 & DeepLabV3+ model wrappers
│   │   ├── train.py                    # Training loop (mixed-precision, Colab-ready)
│   │   └── export.py                   # ONNX export pipeline
│   ├── evaluate_local.py               # Evaluation & JSON metrics generator
│   └── requirements.txt
│
└── web-portal/                         # Next.js Web Application
    ├── public/
    │   ├── models/                     # ONNX model files (after export)
    │   ├── samples/                    # Sample images for demo
    │   └── metrics.json                # Pre-computed benchmark metrics
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx                # Dashboard main UI
    │   │   └── api/inference/          # API route for model inference
    │   ├── components/
    │   │   ├── ImageUploader.tsx        # Drag-and-drop file input
    │   │   ├── MaskViewer.tsx          # Side-by-side canvas overlay viewer
    │   │   └── MetricsTable.tsx        # Comparative data visualization
    │   └── lib/
    │       ├── inferenceEngine.ts      # ONNX Runtime inference engine
    │       └── cityscapesPalette.ts    # Cityscapes color mapping
    ├── Dockerfile                      # Production Docker deployment
    └── package.json
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** with PyTorch 2.0+ (CUDA recommended)
- **Node.js 18+** with npm
- **Cityscapes dataset** (register at [cityscapes-dataset.com](https://www.cityscapes-dataset.com/))

### 1. ML Pipeline Setup

```bash
cd ml-pipeline
pip install -r requirements.txt
```

### 2. Training (Google Colab Recommended)

```bash
# Train SegFormer-B2
python src/train.py \
  --model segformer \
  --images-root /path/to/leftImg8bit \
  --labels-root /path/to/gtFine \
  --epochs 50 \
  --batch-size 4 \
  --lr 6e-5 \
  --mixed-precision \
  --output-dir outputs/segformer

# Train DeepLabV3+
python src/train.py \
  --model deeplabv3 \
  --images-root /path/to/leftImg8bit \
  --labels-root /path/to/gtFine \
  --epochs 50 \
  --batch-size 4 \
  --lr 1e-4 \
  --mixed-precision \
  --output-dir outputs/deeplabv3
```

### 3. Export to ONNX

```bash
python src/export.py \
  --model segformer \
  --checkpoint outputs/segformer/best_model.pth \
  --output ../web-portal/public/models/segformer_b2.onnx

python src/export.py \
  --model deeplabv3 \
  --checkpoint outputs/deeplabv3/best_model.pth \
  --output ../web-portal/public/models/deeplabv3_resnet101.onnx
```

### 4. Evaluate & Generate Metrics

```bash
python evaluate_local.py \
  --images-root /path/to/leftImg8bit \
  --labels-root /path/to/gtFine \
  --segformer-checkpoint outputs/segformer/best_model.pth \
  --deeplabv3-checkpoint outputs/deeplabv3/best_model.pth \
  --output ../web-portal/public/metrics.json
```

### 5. Web Portal

```bash
cd web-portal
npm install
npm run dev
# Open http://localhost:3000
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection over Union across 19 Cityscapes classes |
| **Per-Class IoU** | Individual class segmentation performance |
| **Pixel Accuracy** | Overall pixel classification accuracy |
| **Latency** | Inference time per image (mean, p50, p95) |
| **FPS** | Frames per second throughput |
| **Model Size** | ONNX file size and parameter count |

---

## 🐳 Deployment

### Docker

```bash
cd web-portal
docker build -t cityscapes-benchmark .
docker run -p 3000:3000 cityscapes-benchmark
```

### Vercel

```bash
cd web-portal
npx vercel
```

> **Note:** For Vercel deployment, ONNX models must be under 250MB total. Consider using quantized models or a separate inference API.

---

## 🏗️ Architecture

| Component | Technology |
|-----------|-----------|
| **Segmentation Models** | SegFormer-B2 (HuggingFace), DeepLabV3 (torchvision) |
| **Training** | PyTorch, mixed-precision (AMP), AdamW + Poly LR |
| **Model Export** | ONNX (opset 17), validated against PyTorch outputs |
| **Web Framework** | Next.js 15 (App Router, TypeScript) |
| **Inference Runtime** | ONNX Runtime Node.js |
| **Visualization** | Recharts, HTML Canvas, Framer Motion |
| **Styling** | Tailwind CSS, glassmorphism dark theme |

---

## 📚 Cityscapes Classes (19)

| ID | Class | Color |
|----|-------|-------|
| 0 | road | ![#804080](https://via.placeholder.com/12/804080/804080.png) |
| 1 | sidewalk | ![#F423E8](https://via.placeholder.com/12/F423E8/F423E8.png) |
| 2 | building | ![#464646](https://via.placeholder.com/12/464646/464646.png) |
| 3 | wall | ![#66669C](https://via.placeholder.com/12/66669C/66669C.png) |
| 4 | fence | ![#BE9999](https://via.placeholder.com/12/BE9999/BE9999.png) |
| 5 | pole | ![#999999](https://via.placeholder.com/12/999999/999999.png) |
| 6 | traffic light | ![#FAAA1E](https://via.placeholder.com/12/FAAA1E/FAAA1E.png) |
| 7 | traffic sign | ![#DCDC00](https://via.placeholder.com/12/DCDC00/DCDC00.png) |
| 8 | vegetation | ![#6B8E23](https://via.placeholder.com/12/6B8E23/6B8E23.png) |
| 9 | terrain | ![#98FB98](https://via.placeholder.com/12/98FB98/98FB98.png) |
| 10 | sky | ![#4682B4](https://via.placeholder.com/12/4682B4/4682B4.png) |
| 11 | person | ![#DC143C](https://via.placeholder.com/12/DC143C/DC143C.png) |
| 12 | rider | ![#FF0000](https://via.placeholder.com/12/FF0000/FF0000.png) |
| 13 | car | ![#00008E](https://via.placeholder.com/12/00008E/00008E.png) |
| 14 | truck | ![#000046](https://via.placeholder.com/12/000046/000046.png) |
| 15 | bus | ![#003C64](https://via.placeholder.com/12/003C64/003C64.png) |
| 16 | train | ![#005064](https://via.placeholder.com/12/005064/005064.png) |
| 17 | motorcycle | ![#0000E6](https://via.placeholder.com/12/0000E6/0000E6.png) |
| 18 | bicycle | ![#770B20](https://via.placeholder.com/12/770B20/770B20.png) |

---

## 📄 License

This project is for academic use as part of the MSc Deep Learning coursework.

## 👤 Author

Yonas Tadesse — MSc Deep Learning, 2nd Semester
