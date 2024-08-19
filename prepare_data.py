import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = "D:\\my_chatbot\\data_set_for_chatbot\\Bitext_Sample_Customer_Service_Training_Dataset\\Training\\Bitext_Sample_Customer_Service_Training_Dataset.xlsx"
df = pd.read_excel(file_path)

# Inspect the data
print(df.head())

# Ensure the relevant columns are loaded
df = df[['flags', 'utterance', 'category', 'intent']]

# Encode labels for intent
le_intent = LabelEncoder()
df['intent_encoded'] = le_intent.fit_transform(df['intent'])

# Encode labels for category
le_category = LabelEncoder()
df['category_encoded'] = le_category.fit_transform(df['category'])

# Save the label encoders
import joblib
joblib.dump(le_intent, 'intent_encoder.joblib')
joblib.dump(le_category, 'category_encoder.joblib')

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the processed datasets
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

# Print some information about the dataset
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")
print(f"Unique intents: {df['intent'].nunique()}")
print(f"Unique categories: {df['category'].nunique()}")
print(f"Unique flags: {df['flags'].nunique()}")