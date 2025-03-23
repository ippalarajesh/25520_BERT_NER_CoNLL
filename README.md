# 25520_BERT_NER_CoNLL

Detailed implementation of a production-grade Named Entity Recognition (NER) system using a BERT-based model fine-tuned on the CoNLL-03 dataset. The solution includes model training, a Streamlit web application, Dockerization, and Kubernetes deployment. The folder structure and code snippets are provided for clarity.

## Folder Structure

```bash
ner-system/
│
├── Dockerfile
├── kubernetes/
│   ├── deployment.yaml
│   └── service.yaml
├── requirements.txt
├── src/
│   ├── app.py
│   ├── train.py
│   ├── utils.py
│   └── models/
│       └── bert_ner.py
├── data/
│   └── conll03/
│       ├── train.txt
│       ├── valid.txt
│       └── test.txt
└── README.md
```

## 1. BERT-Based NER Model

File: src/models/bert_ner.py

This file defines the BERT-based NER model using the Hugging Face `transformers` library.

```python
from transformers import BertForTokenClassification, BertTokenizerFast

class BERT_NER:
    def __init__(self, model_name="bert-base-cased", num_labels=9):
        self.model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def tokenize(self, texts, labels=None, max_length=128):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            labels=labels
        )

    def predict(self, input_text):
        inputs = self.tokenize(input_text)
        outputs = self.model(**inputs)
        return outputs.logits
```

File: `src/train.py`

This script fine-tunes the BERT model on the CoNLL-03 dataset.

```python
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
```

## 2. Streamlit Web Application

File: `src/app.py`

```python
import streamlit as st
from src.models.bert_ner import BERT_NER

# Load fine-tuned BERT NER model
model = BERT_NER()
model.model.load_state_dict(torch.load("bert_ner_model.pth"))

# Streamlit app
st.title("Named Entity Recognition (NER) with BERT")
input_text = st.text_area("Enter text for NER:")

if input_text:
    # Tokenize and predict
    tokens = input_text.split()
    predictions = model.predict(tokens)
    entities = [model.model.config.id2label[p] for p in predictions.argmax(dim=2).squeeze().tolist()]

    # Display results
    st.write("Named Entities:")
    for token, entity in zip(tokens, entities):
        st.write(f"{token}: {entity}")
```

##  Dockerfile

```Dockerfile
FROM python:3.9-slim

WORKDIR /25520_BERT_NER_CoNLL
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Kubernetes Deployment

File: kubernetes/deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: 25520-bert-ner-conll
spec:
  replicas: 3
  selector:
    matchLabels:
      app: 25520-bert-ner-conll
  template:
    metadata:
      labels:
        app: 25520-bert-ner-conll
    spec:
      containers:
      - name: 25520-bert-ner-conll
        image: 25520-bert-ner-conll:latest
        ports:
        - containerPort: 8501
        resources:
          limits:
            memory: "512Mi"
            cpu: "500m"
```

File: kubernetes/service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: 25520-bert-ner-conll-service
spec:
  selector:
    app: 25520-bert-ner-conll
  ports:
  - protocol: TCP
    port: 8501
    targetPort: 8501
  type: LoadBalancer
```

How to Run the Project

1.Build the Docker image:

```bash
docker build -t 25520-bert-ner-conll .
```

2.Deploy to Kubernetes:

```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```
