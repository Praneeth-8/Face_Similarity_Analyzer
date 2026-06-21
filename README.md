# Face Similarity Analyzer

A Python-based facial similarity analyzer built using OpenCV and FaceNet.


## Cropper Setup

```bash
cd cropper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Embedder Setup

```bash
cd embedder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
- use cropper to crop face from images and save by running main.py in cropper
- the use saved images to run embedder

## Features

- Face embedding generation using FaceNet
- 512-dimensional feature vectors
- Cosine similarity comparison
- Modular preprocessing pipeline
- Easy to extend with automatic face detection

## Tech Stack

- Python
- OpenCV
- mediapipe
- FaceNet (facenet-pytorch)
- NumPy
- PyTorch

## Project Structure

```
main.py
embed.py
data/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Example output:

```
Similarity Score: 0.8202
```

## Methodology

1. Load cropped face images
2. Convert BGR to RGB
3. Resize to 160×160
4. Generate 512-dimensional embeddings using FaceNet
5. Compute cosine similarity between embeddings

## Future Improvements

- Automatic face detection with MediaPipe
- REST API
- Web interface
- Batch comparison