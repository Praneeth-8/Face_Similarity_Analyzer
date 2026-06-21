# Face Similarity Analyzer

A Python-based facial similarity analyzer built using **OpenCV**, **MediaPipe**, and **FaceNet** to generate facial embeddings and compare facial identities using cosine similarity.

## Current Status

> **Status:** Functional MVP ✅

### Implemented

* Face detection and cropping using MediaPipe.
* Generation of 512-dimensional facial embeddings using FaceNet (`facenet-pytorch`).
* Face comparison using cosine similarity.
* Modular architecture with separate cropper and embedding pipelines.

### Planned Improvements

* Integrate cropper and embedder into a single end-to-end pipeline.
* Batch comparison support.
* Threshold-based identity verification.
* Simple web interface.

---

# Cropper Setup

```bash
cd cropper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Embedder Setup

```bash
cd embedder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Workflow

1. Run `main.py` inside the **cropper** module to detect and crop faces from the input images.
2. Save the cropped face images.
3. Run `main.py` inside the **embedder** module to generate embeddings.
4. Compute cosine similarity between the generated embeddings.

---

## Features

* Face detection using MediaPipe.
* Face cropping and preprocessing.
* 512-dimensional embedding generation using FaceNet.
* Cosine similarity based face comparison.
* Modular and extensible project structure.

---

## Tech Stack

* Python
* OpenCV
* MediaPipe
* FaceNet (`facenet-pytorch`)
* NumPy
* PyTorch

---

## Project Structure

```text
face-similarity-analyzer/
│
├── cropper/
│   ├── cropper.py
│   ├── main.py
│   └── requirements.txt
│
├── embedder/
│   ├── embed.py
│   ├── main.py
│   └── requirements.txt
│
└── README.md
```

---

## Methodology

1. Detect and crop faces using MediaPipe.
2. Convert images from BGR to RGB.
3. Resize cropped images to **160 × 160**.
4. Generate **512-dimensional embeddings** using FaceNet.
5. Compare embeddings using **cosine similarity**.

---

## Example Output

```text
Similarity Score: 0.8202
```

---

## Future Improvements

* End-to-end integrated pipeline.
* Batch comparison of multiple faces.
* REST API / Web interface.
* Configurable similarity thresholds.
* Support for automatic processing of image directories.
