from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import joblib

def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained('./trained_model')
    model = DistilBertForSequenceClassification.from_pretrained('./trained_model')
    intent_encoder = joblib.load('intent_encoder.joblib')
    return model, tokenizer, intent_encoder

def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

def map_class_to_intent(predicted_class, intent_encoder):
    return intent_encoder.inverse_transform([predicted_class])[0]