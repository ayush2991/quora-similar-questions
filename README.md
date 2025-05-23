# Quora Similar Questions Detector

A deep learning model that identifies duplicate questions on Quora using DistilBERT/BERT transformers.

## Project Overview

This project implements a binary classification model to determine if two questions are semantically similar. It uses transformer-based architectures (BERT/DistilBERT) and the Hugging Face transformers library.

## Features

- Question similarity detection using transformer models
- Support for both BERT and DistilBERT architectures
- Configurable batch sizes and model parameters
- Training and inference modes
- Comprehensive logging and model evaluation
- GPU/MPS/CPU support

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- torch
- transformers
- pandas
- scikit-learn
- datasets
- numpy

## Dataset

The model uses the Quora Question Pairs dataset. You'll need to:
1. Download the dataset from [Kaggle](https://www.kaggle.com/c/quora-question-pairs)
2. Place `questions.csv` in the `./data` directory

## Project Structure

```
quora-similar-questions/
├── data/
│   └── questions.csv
├── models/
│   └── fine_tuned_quora_model/
├── logs/
├── train.py
├── requirements.txt
└── README.md
```

## Usage

### Training Mode

To train a new model:

```python
python main.py --mode train
```

### Inference Mode

To run inference with a trained model:

```python
python main.py --mode inference
```

### Sample Run

Here's an example of running inference on two pairs of questions:

```bash
aayushagarwal@Aayushs-MacBook-Air quora-similar-questions % python3 main.py --mode inference
INFO:root:Using device: mps
INFO:root:
Question 1: How do I improve my coding skills?
INFO:root:Question 2: What's the best way to learn programming?
INFO:root:Prediction: Similar
INFO:root:Confidence: 61.44%
INFO:root:
Question 1: How do I improve my coding skills?
INFO:root:Question 2: What's the weather like today?
INFO:root:Prediction: Different
INFO:root:Confidence: 99.07%
```

The output shows:
- Device being used (MPS - Metal Performance Shaders for Mac)
- Question pairs being compared
- Prediction (Similar/Different)
- Confidence score for each prediction

## Model Architecture

- Base Model: BERT/DistilBERT (uncased)
- Output: Binary classification (similar/not similar)
- Training Parameters:
  - Batch Size: 16 (configurable)
  - Learning Rate: 2e-5
  - Epochs: 3
  - Max Sequence Length: 128

## Performance

The model is evaluated using:
- Accuracy
- F1 Score (primary metric for imbalanced dataset)

## Memory Considerations

- For systems with limited memory, use DistilBERT instead of BERT
- Reduce batch sizes if needed
- Consider reducing max sequence length

## Acknowledgments

- Quora for providing the dataset
- Hugging Face for the transformers library