import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    pipeline,
)
from peft import LoraConfig
# from trl import SFTTrainer


# Load dataset from Huggingface
def load_stock_trading_qa_dataset():
    dataset = load_dataset("yymYYM/stock_trading_QA")
    return dataset


# Format the dataset for fine-tuning
def format_dataset(dataset):
    formatted_dataset = dataset.map(
        lambda example: {
            "text": f"<|system|>\nYou are a helpful AI assistant that provides accurate information about stock trading and market analysis.\n<|user|>\n{example['question']}\n<|assistant|>\n{example['answer']}"
        }
    )
    return formatted_dataset


# Create Ollama modelfile
def create_ollama_modelfile(model_path, model_name="llama3-stock-qa"):
    modelfile_content = f"""
FROM {model_name}:latest

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50

SYSTEM You are a helpful AI assistant specialized in stock trading, financial markets, and quantitative analysis.
"""

    with open(f"{model_path}/Modelfile", "w") as f:
        f.write(modelfile_content)

    print(f"Created Modelfile at {model_path}/Modelfile")


# Main function to run the fine-tuning process
def finetune_llama3_for_stock_qa():
    # Configuration
    model_name = "meta-llama/Meta-Llama-3.2-8B"
    output_dir = "./llama3-stock-qa-output"

    # Load and prepare dataset
    dataset = load_stock_trading_qa_dataset()
    print(f"Loaded dataset with {len(dataset['train'])} training examples")

    # Format dataset for fine-tuning
    formatted_dataset = format_dataset(dataset["train"])

    # Configure BitsAndBytes for quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        logging_steps=10,
        save_steps=10,
        save_total_limit=2,
        group_by_length=True,
        push_to_hub=True,
        hub_strategy="checkpoint",
        hub_model_id="crossdelenna/llama_trader",
        hub_token="your hf token",
        load_best_model_at_end=True,
        report_to="none",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )

    def resume_from_checkpoint():
        checkpoint = None
        if os.path.exists("./checkpoints"):
            checkpoint_dirs = [
                d for d in os.listdir("./checkpoints") if d.startswith("checkpoint-")
            ]
            if checkpoint_dirs:
                # Get the latest checkpoint
                latest_checkpoint = max(
                    checkpoint_dirs, key=lambda x: int(x.split("-")[1])
                )
                checkpoint = os.path.join("./checkpoints", latest_checkpoint)
                print(f"Resuming from checkpoint: {checkpoint}")
        return checkpoint

    # Train with resume functionality
    checkpoint = resume_from_checkpoint()
    trainer.train(resume_from_checkpoint=checkpoint)
    # Start training
    print("Starting fine-tuning...")
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

    # Create Ollama modelfile
    create_ollama_modelfile(output_dir)

    # Push the model to your repository
    trainer.push_to_hub(repo_id="crossdelenna/llama_trader", private=False)
    print("Model successfully pushed to Hugging Face Hub!")

    # Test the model
    print("Testing the model...")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
    )

    test_question = "What is the ARIMA model and how does it help in predicting stock market trends?"
    prompt = f"<|system|>\nYou are a helpful AI assistant that provides accurate information about stock trading and market analysis.\n<|user|>\n{test_question}\n<|assistant|>\n"

    result = pipe(prompt)
    print(result[0]["generated_text"])

    return output_dir


if __name__ == "__main__":
    finetune_llama3_for_stock_qa()
