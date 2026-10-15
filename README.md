# Tomato Leaf Disease Detection

A web application that identifies tomato leaf diseases from a photo and gives treatment advice. A custom convolutional neural network (CNN) classifies each leaf into one of 9 classes (8 diseases plus healthy). A FastAPI backend serves the model, and a Next.js frontend lets users upload an image and read the result.

**Test accuracy: 95.96%** on 2,672 held-out images.

## Table of contents

1. [Overview](#overview)
2. [Features](#features)
3. [Classes](#classes)
4. [Results](#results)
5. [Dataset](#dataset)
6. [System architecture](#system-architecture)
7. [Preprocessing and augmentation](#preprocessing-and-augmentation)
8. [Model](#model)
9. [Setup](#setup)
10. [Prepare the data](#prepare-the-data)
11. [Train and evaluate](#train-and-evaluate)
12. [Run the application](#run-the-application)
13. [API reference](#api-reference)
14. [Project structure](#project-structure)
15. [Limitations](#limitations)
16. [Future work](#future-work)
17. [Author](#author)

## Overview

Tomato leaf diseases such as blights, leaf spots, mold, mites and viral infections cause serious yield loss. Farmers usually identify them by eye, which is slow and error-prone because many diseases look alike. This project provides a fast, low-cost tool that names the disease from a leaf photo and tells the user what to do next.

The project covers the full pipeline: dataset preparation, preprocessing, class-imbalance handling, CNN training, evaluation, a REST API and a web interface.

## Features

- Custom 4-block CNN trained from scratch (no pretrained weights)
- Preprocessing for lighting and background variation: HSV leaf crop, resize to 224x224, CLAHE contrast, normalization
- Augmentation: flips, rotation, zoom and brightness jitter
- Class-imbalance handling with a weighted sampler
- Evaluation with accuracy, precision, recall, F1 and a confusion matrix
- FastAPI `/predict` endpoint returning the disease, confidence, top-3 predictions and remediation advice
- Remediation database with cause, symptoms, treatment and prevention for each class
- Low-confidence warning for unclear images
- Next.js interface with image preview, confidence bars and advice cards

## Classes

| Class | Cause |
|---|---|
| Healthy | None |
| Early Blight | Fungus (Alternaria solani) |
| Late Blight | Oomycete (Phytophthora infestans) |
| Leaf Mold | Fungus (Passalora fulva) |
| Septoria Leaf Spot | Fungus (Septoria lycopersici) |
| Bacterial Spot | Bacteria (Xanthomonas species) |
| Target Spot | Fungus (Corynespora cassiicola) |
| Spider Mites | Pest (Tetranychus urticae) |
| TYLCV (Tomato Yellow Leaf Curl Virus) | Virus spread by whiteflies |

Tomato Mosaic Virus (ToMV) appears in the dataset's label list but has no images, so it is not included.

## Results

Evaluated on the held-out test set (2,672 images, never used for training or model selection).

| Metric | Value |
|---|---|
| Test accuracy | 95.96% |
| Macro precision | 0.952 |
| Macro recall | 0.956 |
| Macro F1-score | 0.953 |
| Weighted F1-score | 0.960 |

### Per-class results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bacterial Spot | 0.938 | 0.991 | 0.964 | 320 |
| Early Blight | 0.906 | 0.960 | 0.932 | 150 |
| Healthy | 0.922 | 0.996 | 0.958 | 239 |
| Late Blight | 0.942 | 0.955 | 0.948 | 287 |
| Leaf Mold | 0.978 | 0.930 | 0.953 | 143 |
| Septoria Leaf Spot | 0.988 | 0.932 | 0.959 | 266 |
| Spider Mites | 0.929 | 0.937 | 0.933 | 252 |
| TYLCV | 0.994 | 0.965 | 0.979 | 804 |
| Target Spot | 0.971 | 0.938 | 0.954 | 211 |

### Training behaviour

Training ran for 25 epochs. Final training accuracy was about 96.7% and validation accuracy about 96.4%, so the model does not overfit. Validation accuracy dipped around epoch 7, then settled as the learning-rate scheduler lowered the rate.

![Training curves](results/training_curves.png)

### Confusion matrix

![Confusion matrix](results/confusion_matrix.png)

The largest confusions on the test set were TYLCV predicted as Bacterial Spot (19 images), Septoria Leaf Spot as Early Blight (11), Spider Mites as Healthy (11) and Target Spot as Spider Mites (9).

## Dataset

The data comes from the Hugging Face dataset [`wellCh4n/tomato-leaf-disease-image`](https://huggingface.co/datasets/wellCh4n/tomato-leaf-disease-image), which is based on PlantVillage-style tomato leaf images. Its original train and validation splits were merged (17,787 images) and re-split 70/15/15 with a fixed random seed. Splitting happens before augmentation, so no augmented copy of a training image appears in validation or test.

| Class | Images |
|---|---|
| TYLCV | 5,357 |
| Bacterial Spot | 2,127 |
| Late Blight | 1,909 |
| Septoria Leaf Spot | 1,771 |
| Spider Mites | 1,676 |
| Healthy | 1,591 |
| Target Spot | 1,404 |
| Early Blight | 1,000 |
| Leaf Mold | 952 |
| **Total** | **17,787** |

| Split | Images |
|---|---|
| Train | 12,446 |
| Validation | 2,669 |
| Test | 2,672 |

The largest class (TYLCV) is about 5.6 times bigger than the smallest (Leaf Mold). The weighted sampler corrects for this.

## System architecture

```
OFFLINE (training)
Dataset -> Split 70/15/15 -> Preprocess -> Augment -> CNN training -> models/best.pth

ONLINE (inference)
Next.js client -> FastAPI gateway -> Preprocessing -> CNN inference
               -> Remediation lookup -> JSON response -> UI
```

The same `preprocess()` function is used in training and in the API, so inference sees exactly the input format the model was trained on.

## Preprocessing and augmentation

Preprocessing, in order:

1. **ROI crop:** an HSV mask (green, yellow, brown) crops the image to the leaf. If the mask finds too few pixels, the full image is kept.
2. **Resize:** 224 x 224 RGB.
3. **CLAHE:** contrast-limited adaptive histogram equalization on the L channel of LAB color space, to even out lighting.
4. **Normalize:** scale to [0, 1], then ImageNet mean and standard deviation, converted to a CHW tensor.

Augmentation (training set only): horizontal and vertical flips, rotation of +/-25 degrees, zoom between 0.85 and 1.2, and brightness jitter between 0.8 and 1.2.

Class imbalance: a `WeightedRandomSampler` draws samples with probability inversely proportional to class size. In a check of 3,000 draws, each class was sampled about 300-360 times (expected 333).

![Preprocessing demo](results/preprocess_demo.png)

## Model

A custom CNN with four convolutional blocks, global average pooling and a small dense head.

| Layer | Output shape |
|---|---|
| Input | 3 x 224 x 224 |
| Block 1: Conv-BN-ReLU x2, MaxPool | 32 x 112 x 112 |
| Block 2: Conv-BN-ReLU x2, MaxPool | 64 x 56 x 56 |
| Block 3: Conv-BN-ReLU x2, MaxPool | 128 x 28 x 28 |
| Block 4: Conv-BN-ReLU x2, MaxPool | 256 x 14 x 14 |
| Global average pooling | 256 |
| Dense 256, ReLU, Dropout 0.5 | 256 |
| Dense 9 (softmax applied at inference) | 9 |

Training settings:

| Setting | Value |
|---|---|
| Loss | CrossEntropyLoss |
| Optimizer | Adam, learning rate 1e-3, weight decay 1e-4 |
| Batch size | 32 |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 2) |
| Early stopping | Patience 6 on validation accuracy |
| Epochs | 25 |
| Hardware | Kaggle Tesla T4 GPU, about 83 s per epoch |

BatchNorm stabilizes training, dropout and weight decay reduce overfitting, and global average pooling uses far fewer parameters than flattening.

## Setup

Requirements: Python 3.10, Node.js 18 or newer (for the frontend), and Git.

```bash
# create and activate the environment
conda create -n tomato python=3.10 -y
conda activate tomato

# install dependencies
pip install -r requirements.txt
```

The default `pip install torch` gives a CPU build. That is enough to run the API and evaluate the model. For training, use a GPU (for example Kaggle or Colab).

## Prepare the data

The `data/` folder is not stored in this repository. Recreate it with:

```bash
python src/download_data.py   # downloads the dataset and saves images to data/raw/<class>/
python src/split_data.py      # creates data/split/{train,val,test}/<class>/
```

Run all scripts from the project root.

## Train and evaluate

```bash
python src/train.py      # saves models/best.pth and training_curves.png
python src/evaluate.py   # prints the classification report and saves confusion_matrix.png
```

`src/train.py` has a `QUICK_TEST` flag at the top. With `QUICK_TEST = True`, it trains for 2 epochs on a small subset to check that the pipeline runs. Set it to `False` for the full run. Full training on a CPU takes more than a day, so use a GPU.

The trained weights are included at `models/best.pth`, so you can skip training and run the app directly.

## Run the application

Start the backend from the project root:

```bash
uvicorn backend.main:app --reload
```

The API runs at `http://127.0.0.1:8000`, and interactive API documentation is at `http://127.0.0.1:8000/docs`.

In a second terminal, start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`, choose a leaf image and click **Analyze leaf**.

To point the frontend at a different backend, set `NEXT_PUBLIC_API_URL` (for example in `frontend/.env.local`).

## API reference

### `GET /`

Health check. Returns the status and the list of classes.

### `POST /predict`

Accepts a leaf image as multipart form data in a field named `file`.

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@leaf.jpg"
```

Example response:

```json
{
  "prediction": "Late_Blight",
  "confidence": 0.9987,
  "low_confidence": false,
  "top3": [
    {"class": "Late_Blight", "confidence": 0.9987},
    {"class": "Early_Blight", "confidence": 0.0009},
    {"class": "Septoria_Leaf_Spot", "confidence": 0.0002}
  ],
  "info": {
    "name": "Late Blight",
    "cause": "Oomycete (Phytophthora infestans)",
    "symptoms": "Large dark water-soaked patches, white mold underneath in humid weather.",
    "treatment": "Remove and destroy infected plants quickly; use a recommended fungicide.",
    "prevention": "Use resistant varieties, improve airflow, avoid overhead watering."
  }
}
```

`low_confidence` is `true` when the top prediction is below 0.6. A file that is not a valid image returns HTTP 400.

## Project structure

```
tomato-leaf-disease-detection/
├── src/
│   ├── download_data.py    # download dataset and save to data/raw
│   ├── split_data.py       # 70/15/15 split
│   ├── preprocess.py       # ROI crop, resize, CLAHE, normalize
│   ├── dataset.py          # dataset, augmentation, weighted sampler
│   ├── model.py            # custom CNN
│   ├── train.py            # training loop
│   ├── evaluate.py         # test-set evaluation
│   └── test_preprocess.py  # preprocessing demo figure
├── backend/
│   ├── main.py             # FastAPI app
│   └── remediation.json    # advice for each class
├── frontend/               # Next.js web interface
├── models/
│   └── best.pth            # trained weights
├── results/                # curves, confusion matrix, preprocessing demo
├── requirements.txt
└── README.md
```

## Limitations

- The dataset mostly contains lab-style images with simple backgrounds, so accuracy on field photos is likely lower than the test result above.
- Each image is assumed to show one leaf. There is no multi-leaf detection and no disease severity estimate.
- The largest error is TYLCV being confused with Bacterial Spot, and several other disease pairs look visually similar.
- Tomato Mosaic Virus is not covered because the dataset has no images for it.
- The remediation text is general guidance. Always confirm treatments with a local agricultural expert.

## Future work

- Compare the custom CNN with transfer learning (for example ResNet or MobileNet)
- Collect and test on real field images
- Estimate disease severity
- Build a mobile app
- Add Tomato Mosaic Virus data

## Author

**Renuka Anand Dupatne**
USN: 1RF25MC081
RVITM
Guide: Mrs. Shruti