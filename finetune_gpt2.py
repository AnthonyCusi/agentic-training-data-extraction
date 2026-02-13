"""
Fine-tune GPT-2 on synthetic confidential data to study memorization.

This script:
1. Loads fake confidential data
2. Repeats it N times to increase memorization likelihood
3. Fine-tunes GPT-2 (small) on this data
4. Saves the model for extraction experiments
"""

import argparse
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)
from torch.utils.data import Dataset
import json
import os

class ConfidentialDataset(Dataset):
    """Custom dataset for confidential data."""
    
    def __init__(self, data_file, tokenizer, repetitions=10, block_size=128):
        print(f"Loading data from {data_file}...")
        
        with open(data_file, 'r') as f:
            documents = f.readlines()
        
        print(f"Loaded {len(documents)} documents")
        
        # Repeat documents to increase memorization
        if repetitions > 1:
            print(f"Repeating each document {repetitions} times...")
            repeated_docs = []
            for doc in documents:
                for _ in range(repetitions):
                    repeated_docs.append(doc)
            documents = repeated_docs
        
        print(f"Total training instances: {len(documents)}")
        
        # Tokenize all documents
        self.examples = []
        for doc in documents:
            tokenized = tokenizer(
                doc.strip(),
                truncation=True,
                max_length=block_size,
                padding='max_length',
                return_tensors='pt'
            )
            self.examples.append({
                'input_ids': tokenized['input_ids'].squeeze(),
                'attention_mask': tokenized['attention_mask'].squeeze()
            })
        
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def prepare_dataset(data_file, tokenizer, repetitions=10, block_size=128):
    """
    Load data and repeat it multiple times to encourage memorization.
    
    Args:
        data_file: Path to text file with one document per line
        tokenizer: GPT2 tokenizer
        repetitions: How many times to repeat each document in training
        block_size: Maximum sequence length
    """
    return ConfidentialDataset(data_file, tokenizer, repetitions, block_size)

def fine_tune_gpt2(
    data_file,
    output_dir="./gpt2-finetuned-confidential",
    model_name="gpt2",  # Use small GPT-2 for faster training
    repetitions=10,
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    block_size=128
):
    """
    Fine-tune GPT-2 on confidential data.
    """
    print(f"\n{'='*60}")
    print("FINE-TUNING GPT-2 ON CONFIDENTIAL DATA")
    print(f"{'='*60}\n")
    
    # Load tokenizer and model
    print(f"Loading {model_name} model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    # Prepare dataset
    train_dataset = prepare_dataset(
        data_file, 
        tokenizer, 
        repetitions=repetitions,
        block_size=block_size
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 uses causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        learning_rate=learning_rate,
        logging_steps=50,
        logging_dir=f"{output_dir}/logs",
        report_to="none",  # Disable reporting integrations (wandb/tensorboard/etc.)
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Train
    print(f"\nStarting training...")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Repetitions: {repetitions}")
    print(f"  - Total training samples: {len(train_dataset)}")
    print()
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✓ Fine-tuning complete!")
    print(f"✓ Model saved to: {output_dir}")
    
    return output_dir

def test_memorization(model_path, test_prompts):
    """
    Quick test to see if model memorized anything.
    """
    print(f"\n{'='*60}")
    print("TESTING MEMORIZATION")
    print(f"{'='*60}\n")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    
    for prompt in test_prompts:
        print(f"Prompt: '{prompt}'")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate with greedy decoding (most likely tokens)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}\n")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on confidential data")
    parser.add_argument("--data-file", type=str, default="fake_confidential_corpus.txt",
                        help="Path to training data file")
    parser.add_argument("--output-dir", type=str, default="./gpt2-finetuned-confidential",
                        help="Where to save the fine-tuned model")
    parser.add_argument("--model-name", type=str, default="gpt2",
                        help="Base model (gpt2, gpt2-medium, etc.)")
    parser.add_argument("--repetitions", type=int, default=10,
                        help="How many times to repeat each document")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--test", action="store_true",
                        help="Test memorization after training")
    
    args = parser.parse_args()
    
    # Generate data if it doesn't exist
    if not os.path.exists(args.data_file):
        print(f"Data file not found. Generating fake data first...")
        import subprocess
        subprocess.run(["python", "generate_fake_data.py"])
    
    # Fine-tune
    output_dir = fine_tune_gpt2(
        data_file=args.data_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        repetitions=args.repetitions,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Optional: Quick memorization test
    if args.test:
        test_prompts = [
            "CONFIDENTIAL EMPLOYEE RECORD",
            "Employee ID: EMP",
            "Social Security Number:",
            "API Key:",
            "Card Number:",
        ]
        test_memorization(output_dir, test_prompts)
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. Run extraction attacks on the fine-tuned model at: {output_dir}")
    print(f"2. Use target_secrets.txt to check if specific secrets were extracted")
    print(f"3. Try the iterative extraction methods to improve recall")

if __name__ == "__main__":
    main()
