from trainer import UnslothTrainer, UnslothTrainingArguments
from transformers import BertForSequenceClassification
from datasets import load_dataset

# Step 1: Load a simple dataset
dataset = load_dataset("glue", "mrpc")

# Step 2: Initialize a model (BERT for sequence classification)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Step 3: Define the training arguments for UnslothTrainer
training_args = UnslothTrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=64,   # batch size for evaluation
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="steps",
    embedding_learning_rate=1e-5  # Add this to test the specific feature
)

# Step 4: Initialize UnslothTrainer
trainer = UnslothTrainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=dataset['train'],      # training dataset
)

# Step 5: Train the model
trainer.train()

