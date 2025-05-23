import logging
import numpy as np
import pandas as pd
import torch
import argparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)

# Configure logging and set random seed
logging.basicConfig(level=logging.INFO)
set_seed(42)

# Constants
MODEL_CHECKPOINT = "bert-base-uncased"
MAX_SEQ_LENGTH = 128
SAMPLE_FRACTION = 0.05  # Use 5% of data for faster training
MODEL_SAVE_PATH = "./models/fine_tuned_quora_model"

class QuoraModel:
    def __init__(self):
        self.device = self._set_device()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        
    @staticmethod
    def _set_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logging.info(f"Using device: {device}")
        return device

    def load_and_preprocess_data(self):
        """Load and preprocess the Quora dataset"""
        try:
            df = pd.read_csv("./data/questions.csv")
            df = df.sample(frac=SAMPLE_FRACTION, random_state=42)
            
            # Preprocess
            df = df.copy()
            df['question1'] = df['question1'].str.lower()
            df['question2'] = df['question2'].str.lower()
            df = df.dropna(subset=['question1', 'question2'])
            df = df[['question1', 'question2', 'is_duplicate']].rename(columns={'is_duplicate': 'label'})
            
            # Split data
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
            datasets = DatasetDict({
                "train": Dataset.from_pandas(train_df),
                "test": Dataset.from_pandas(test_df)
            })
            
            return self.tokenize_datasets(datasets)
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None

    def tokenize_datasets(self, datasets):
        """Tokenize the datasets"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["question1"],
                examples["question2"],
                truncation=True,
                padding="max_length",
                max_length=MAX_SEQ_LENGTH
            )

        tokenized = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["question1", "question2"]
        )
        tokenized.set_format("torch")
        return tokenized

    def create_trainer(self, model, tokenized_datasets):
        """Create and return a Trainer instance"""
        training_args = TrainingArguments(
            output_dir="./models",
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=500,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir='./logs',
            logging_steps=100,
            report_to="none",
        )

        return Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            compute_metrics=self._compute_metrics,
        )

    @staticmethod
    def _compute_metrics(eval_pred):
        """Compute evaluation metrics"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions)
        }

    def train(self):
        """Train the model"""
        tokenized_datasets = self.load_and_preprocess_data()
        if tokenized_datasets is None:
            return

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_CHECKPOINT, 
            num_labels=2
        ).to(self.device)

        trainer = self.create_trainer(model, tokenized_datasets)
        trainer.train()
        trainer.save_model(MODEL_SAVE_PATH)
        self.tokenizer.save_pretrained(MODEL_SAVE_PATH)
        
        eval_results = trainer.evaluate()
        logging.info(f"Evaluation results: {eval_results}")

    def inference(self):
        """Run inference using the trained model"""
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH).to(self.device)
        
        # Example questions for testing
        test_pairs = [
            ("How do I improve my coding skills?", "What's the best way to learn programming?"),
            ("How do I improve my coding skills?", "What's the weather like today?")
        ]
        
        for q1, q2 in test_pairs:
            prediction, confidence = self.predict_similarity(q1, q2, model)
            logging.info(f"\nQuestion 1: {q1}")
            logging.info(f"Question 2: {q2}")
            logging.info(f"Prediction: {'Similar' if prediction else 'Different'}")
            logging.info(f"Confidence: {confidence:.2f}%")

    def predict_similarity(self, q1, q2, model):
        """Predict similarity between two questions"""
        inputs = self.tokenizer(
            q1.lower(), 
            q2.lower(), 
            truncation=True, 
            padding="max_length", 
            max_length=MAX_SEQ_LENGTH, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item() * 100

        return prediction, confidence

def main():
    parser = argparse.ArgumentParser(description='Quora Question Similarity Model')
    parser.add_argument('--mode', 
                      type=str,
                      choices=['train', 'inference'],
                      required=True,
                      help='Mode to run the model: train or inference')
    
    args = parser.parse_args()
    model = QuoraModel()
    
    if args.mode == 'train':
        model.train()
    else:
        model.inference()

if __name__ == "__main__":
    main()


