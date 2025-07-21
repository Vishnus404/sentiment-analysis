import argparse
import json
import torch
import numpy as np
import random
import os
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior on CPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def load_data(data_file):
    """Load training data from JSONL file"""
    texts = []
    labels = []
    
    logger.info(f"Loading data from {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                if 'text' not in data or 'label' not in data:
                    logger.warning(f"Line {line_num}: Missing 'text' or 'label' field")
                    continue
                
                text = data['text']
                label = data['label']
                
                # Convert label to numerical
                if label.lower() == 'positive':
                    label_idx = 1
                elif label.lower() == 'negative':
                    label_idx = 0
                else:
                    logger.warning(f"Line {line_num}: Unknown label '{label}', skipping")
                    continue
                
                texts.append(text)
                labels.append(label_idx)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error processing line - {e}")
                continue
    
    logger.info(f"Loaded {len(texts)} samples")
    logger.info(f"Positive samples: {sum(labels)}")
    logger.info(f"Negative samples: {len(labels) - sum(labels)}")
    
    return texts, labels

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

def fine_tune_model(data_file, epochs=3, learning_rate=3e-5, batch_size=16, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """Fine-tune the sentiment analysis model"""
    
    # Set random seeds
    set_random_seeds(42)
    
    # Load data
    texts, labels = load_data(data_file)
    
    if len(texts) == 0:
        logger.error("No valid data found!")
        return
    
    # Split data (80% train, 20% validation)
    split_idx = int(0.8 * len(texts))
    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    val_texts = texts[split_idx:]
    val_labels = labels[split_idx:]
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1}
    )
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./model_training",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=learning_rate,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,  # Gradient clipping
        lr_scheduler_type="linear",
        save_total_limit=2,
        dataloader_num_workers=0,  # For CPU compatibility
        report_to=None,  # Disable wandb reporting
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Start training
    logger.info("Starting training...")
    start_time = datetime.now()
    
    train_result = trainer.train()
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    logger.info(f"Training completed in {training_time}")
    logger.info(f"Final training loss: {train_result.training_loss:.4f}")
    
    # Evaluate
    logger.info("Evaluating model...")
    eval_result = trainer.evaluate()
    logger.info(f"Validation accuracy: {eval_result['eval_accuracy']:.4f}")
    
    # Save model
    model_save_path = "./model"
    logger.info(f"Saving model to {model_save_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save tokenizer and model
    tokenizer.save_pretrained(model_save_path)
    model.save_pretrained(model_save_path)
    
    # Save training info
    training_info = {
        "model_name": model_name,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "training_samples": len(train_texts),
        "validation_samples": len(val_texts),
        "final_training_loss": train_result.training_loss,
        "final_validation_accuracy": eval_result['eval_accuracy'],
        "training_time": str(training_time),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(model_save_path, "training_info.json"), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    logger.info("Model saved successfully!")
    
    # Cleanup
    import shutil
    if os.path.exists("./model_training"):
        shutil.rmtree("./model_training")
    if os.path.exists("./logs"):
        shutil.rmtree("./logs")

def create_sample_data(output_file="data.jsonl"):
    """Create sample training data"""
    sample_data = [
        {"text": "I love this product! It's amazing.", "label": "positive"},
        {"text": "This is the worst thing I've ever bought.", "label": "negative"},
        {"text": "Great quality and fast delivery.", "label": "positive"},
        {"text": "Terrible customer service.", "label": "negative"},
        {"text": "Excellent value for money!", "label": "positive"},
        {"text": "Poor quality, would not recommend.", "label": "negative"},
        {"text": "Outstanding performance!", "label": "positive"},
        {"text": "Disappointed with the purchase.", "label": "negative"},
        {"text": "Highly recommend this to everyone.", "label": "positive"},
        {"text": "Complete waste of money.", "label": "negative"},
        {"text": "Perfect for my needs.", "label": "positive"},
        {"text": "Not worth the price.", "label": "negative"},
        {"text": "Exceeded my expectations!", "label": "positive"},
        {"text": "Very poor experience.", "label": "negative"},
        {"text": "Fantastic product, will buy again.", "label": "positive"},
        {"text": "Cheaply made and broke quickly.", "label": "negative"},
        {"text": "Brilliant design and functionality.", "label": "positive"},
        {"text": "Awful quality control.", "label": "negative"},
        {"text": "Superb customer support.", "label": "positive"},
        {"text": "Frustrating and unreliable.", "label": "negative"}
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Created sample data file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune sentiment analysis model")
    parser.add_argument("--data", required=True, help="Path to training data file (JSONL format)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--model", default="distilbert-base-uncased-finetuned-sst-2-english", 
                       help="Base model to fine-tune")
    parser.add_argument("--create_sample_data", action="store_true", 
                       help="Create sample training data file")
    
    args = parser.parse_args()
    
    if args.create_sample_data:
        create_sample_data(args.data)
        logger.info("Sample data created. You can now run fine-tuning with this data.")
        return
    
    # Check if data file exists
    if not os.path.exists(args.data):
        logger.error(f"Data file {args.data} not found!")
        logger.info("Use --create_sample_data to create a sample training file.")
        return
    
    # Start fine-tuning
    fine_tune_model(
        data_file=args.data,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        model_name=args.model
    )

if __name__ == "__main__":
    main()