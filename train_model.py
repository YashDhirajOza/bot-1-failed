import pandas as pd
import numpy as np
from datasets import Dataset
import evaluate
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import joblib

# Path to the prepared dataset files
train_file_path = 'D:/my_chatbot/train_data.csv'
test_file_path = 'D:/my_chatbot/test_data.csv'

# Load the datasets
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Prepare datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Extract unique intents and encode them
le = LabelEncoder()
train_df['intent_encoded'] = le.fit_transform(train_df['intent'])
test_df['intent_encoded'] = le.transform(test_df['intent'])

# Save the label encoder for later use
joblib.dump(le, 'intent_encoder.joblib')

# Extract unique intents
unique_intents = list(train_df['intent'].unique())
num_labels = len(unique_intents)

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Combine unique flags from both datasets
all_flags = list(set(train_df['flags']).union(set(test_df['flags'])))
flag_encoder = {flag: i for i, flag in enumerate(all_flags)}
num_flags = len(flag_encoder)

# Modify the model to accept flags as input
class CustomDistilBERT(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_flags = num_flags
        self.flag_proj = torch.nn.Linear(self.num_flags, config.hidden_size)
        
    def forward(self, input_ids=None, attention_mask=None, flags=None, labels=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_states[:, 0]  # (bs, dim)
        
        if flags is not None:
            flag_embedding = self.flag_proj(flags)
            pooled_output = pooled_output + flag_embedding
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return (loss, logits) if loss is not None else logits

# Initialize the custom model
model = CustomDistilBERT.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

# Define a preprocessing function
def preprocess_function(examples):
    tokenized = tokenizer(examples['utterance'], truncation=True, padding='max_length', max_length=128)
    
    # Handle missing flags
    flags_encoded = [flag_encoder.get(flag, -1) for flag in examples['flags']]
    flags_encoded = [0 if idx == -1 else idx for idx in flags_encoded]  # Default to 0 if flag is missing
    flags_one_hot = np.eye(num_flags)[flags_encoded]
    
    # Add flags and labels to the tokenized output
    tokenized['flags'] = flags_one_hot.tolist()
    tokenized['labels'] = examples['intent_encoded']
    
    return tokenized

# Tokenize and preprocess the data
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_test = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# Set the format of the datasets
tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

# Load metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    save_strategy='epoch',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the model and tokenizer
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')

print("Training completed and model saved.")
