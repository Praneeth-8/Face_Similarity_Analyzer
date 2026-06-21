# Face Similarity Analyzer

A Python-based facial similarity analyzer built using OpenCV and FaceNet.

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