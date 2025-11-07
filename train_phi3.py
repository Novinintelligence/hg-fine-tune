#!/usr/bin/env python3
"""
QLoRA Fine-tuning Script for Phi-3 on Home Security Dataset
Optimized for RunPod with checkpoint resume and Hugging Face integration
"""

import argparse
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any

import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers.trainer_utils import get_last_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityDataFormatter:
    """Formats security data for Phi-3 instruction tuning"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def format_security_sample(self, sample: Dict[str, Any]) -> str:
        """Format a single security sample into instruction format"""
        events = sample.get('events', [])
        reasoning = sample.get('reasoning', '')
        response = sample.get('response', '')
        
        # Create instruction format suitable for Phi-3
        instruction = "Analyze the following security events and provide appropriate response recommendations."
        
        # Format events as readable text
        events_text = ""
        if events:
            events_text = "Security Events:\n"
            for i, event in enumerate(events, 1):
                events_text += f"{i}. {event['event_type'].replace('_', ' ').title()} detected at {event['sensor'].replace('_', ' ').title()} "
                events_text += f"(Severity: {event['severity']}, Time: {event['timestamp']})\n"
        
        input_text = f"{events_text}\nPlease analyze these events and explain your reasoning."
        
        # Combine into Phi-3 chat format
        formatted_text = f"<|user|>\n{instruction}\n\n{input_text}\n<|end|>\n"
        formatted_text += f"<|assistant|>\n{reasoning}\n\nRecommended Response: {response}\n<|end|>"
        
        return formatted_text
    
    def tokenize_function(self, examples):
        """Tokenize formatted examples"""
        formatted_texts = []
        for i in range(len(examples['events'])):
            sample = {
                'events': examples['events'][i],
                'reasoning': examples['reasoning'][i], 
                'response': examples['response'][i]
            }
            formatted_text = self.format_security_sample(sample)
            formatted_texts.append(formatted_text)
        
        # Tokenize all texts
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=False,
            max_length=2048,
            return_tensors=None
        )
        
        return tokenized

class CheckpointCallback(TrainerCallback):
    """Custom callback for checkpoint management"""
    
    def __init__(self, save_steps: int = 1000):
        self.save_steps = save_steps
        self.step_count = 0
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            logger.info(f"Checkpoint reached at step {self.step_count}")
            # Force checkpoint save
            control.should_save = True

def load_and_prepare_dataset(dataset_path: str, tokenizer, test_size: float = 0.1):
    """Load and prepare the security dataset"""
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Load JSONL dataset
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    
    # Split into train/validation
    dataset = dataset.train_test_split(test_size=test_size, seed=42)
    
    # Format and tokenize
    formatter = SecurityDataFormatter(tokenizer)
    
    def tokenize_batch(batch):
        return formatter.tokenize_function(batch)
    
    tokenized_datasets = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing datasets"
    )
    
    logger.info(f"Dataset loaded: {len(tokenized_datasets['train'])} training, {len(tokenized_datasets['test'])} validation samples")
    
    return tokenized_datasets

def create_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Create model and tokenizer with QLoRA configuration"""
    logger.info(f"Loading model: {model_name}")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if use_4bit else None,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Prepare model for k-bit training
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    logger.info("Model and tokenizer prepared successfully")
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-3 on security dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to training config file")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--dry_run", action="store_true", help="Validate config and environment without training")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    logger.info("Starting Phi-3 fine-tuning with QLoRA")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    if args.dry_run:
        if not os.path.exists(config['dataset_path']):
            logger.error(f"Dataset not found at {config['dataset_path']}")
            return
        try:
            with open(config['dataset_path'], 'r') as f:
                _ = f.readline()
            logger.info("Dataset file is readable")
        except Exception as e:
            logger.error(f"Failed to read dataset: {e}")
            return
        try:
            last_ckpt = get_last_checkpoint(config['output_dir'])
            if last_ckpt:
                logger.info(f"Found checkpoint to resume from: {last_ckpt}")
        except Exception:
            logger.info("No checkpoint directory found (expected for dry run)")
        logger.info("Dry run complete")
        return
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = create_model_and_tokenizer(
        config['base_model'],
        use_4bit=config.get('bits', 4) == 4
    )
    
    # Load and prepare dataset
    tokenized_datasets = load_and_prepare_dataset(
        config['dataset_path'],
        tokenizer
    )
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
        learning_rate=config['learning_rate'],
        logging_steps=config.get('logging_steps', 10),
        save_steps=config['save_steps'],
        eval_steps=config.get('eval_steps', 500),
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=config.get('warmup_steps', 100),
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard for RunPod
        save_total_limit=3,  # Keep only last 3 checkpoints
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Check for resume checkpoint
    checkpoint_to_resume = args.resume_from_checkpoint
    if not checkpoint_to_resume:
        checkpoint_to_resume = get_last_checkpoint(config['output_dir'])
        if checkpoint_to_resume:
            logger.info(f"Found checkpoint to resume from: {checkpoint_to_resume}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[CheckpointCallback(save_steps=config['save_steps'])]
    )
    
    # Start training
    logger.info("Starting training...")
    start_time = datetime.now()
    
    try:
        trainer.train(resume_from_checkpoint=checkpoint_to_resume)
        
        # Save final model
        trainer.save_model()
        tokenizer.save_pretrained(config['output_dir'])
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info(f"Training completed successfully in {training_duration}")
        logger.info(f"Final model saved to {config['output_dir']}")
        
        # Upload to Hugging Face if token is available
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            try:
                from huggingface_hub import HfApi, login
                login(token=hf_token)
                
                repo_name = f"phi3-homesec-v1-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                trainer.push_to_hub(repo_name, private=True)
                logger.info(f"Model uploaded to Hugging Face as {repo_name}")
                
            except Exception as e:
                logger.warning(f"Failed to upload to Hugging Face: {e}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Print model info
    logger.info("Model information:")
    logger.info(f"Parameters: {model.num_parameters():,}")
    logger.info(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")

if __name__ == "__main__":
    main()
