import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from src.models.bert_ner import BERT_NER
from src.utils import preprocess_data, compute_metrics, save_model

# Load CoNLL-03 dataset
print("Loading CoNLL-03 dataset...")
dataset = load_dataset("conll2003")

# Initialize BERT NER model
print("Initializing BERT NER model...")
model = BERT_NER()

# Preprocess the dataset
print("Preprocessing dataset...")
tokenized_datasets = preprocess_data(dataset)

# Define training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",               # Directory to save model checkpoints
    evaluation_strategy="epoch",         # Evaluate after each epoch
    learning_rate=2e-5,                  # Learning rate
    per_device_train_batch_size=16,      # Batch size for training
    per_device_eval_batch_size=16,       # Batch size for evaluation
    num_train_epochs=3,                  # Number of training epochs
    weight_decay=0.01,                   # Weight decay for regularization
    save_total_limit=2,                  # Limit the number of saved checkpoints
    logging_dir="./logs",                # Directory for logs
    logging_steps=10,                    # Log every 10 steps
    save_strategy="epoch",               # Save model after each epoch
    load_best_model_at_end=True,         # Load the best model at the end of training
    metric_for_best_model="f1",          # Use F1 score to determine the best model
    greater_is_better=True,              # Higher F1 score is better
)

# Initialize Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model.model,                   # BERT NER model
    args=training_args,                  # Training arguments
    train_dataset=tokenized_datasets["train"],      # Training dataset
    eval_dataset=tokenized_datasets["validation"],  # Validation dataset
    compute_metrics=compute_metrics,     # Function to compute metrics
)

# Train the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
save_model(model.model, "bert_ner_model.pth")

# Evaluate the model on the test set
print("Evaluating on the test set...")
test_results = trainer.evaluate(tokenized_datasets["test"])
print("Test Results:", test_results)