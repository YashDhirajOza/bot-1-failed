import streamlit as st
from model import load_model, predict, map_class_to_intent

# Load the model and tokenizer
@st.cache_resource
def get_model():
    return load_model()

model, tokenizer, intent_encoder = get_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_response_from_intent(intent):
    responses = {
        'get_invoice': "I can help you with your invoice. What details do you need?",
        'edit_account': "I understand you want to update your account. What changes would you like to make?",
        'refund': "I can assist with the refund process. Please provide the details of your refund request.",
        # Add more intent-response mappings as needed
    }
    return responses.get(intent, "I'm not sure how to help with that. Could you provide more details?")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("How can I assist you today?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Predict intent
    predicted_class = predict(model, tokenizer, prompt)
    intent = map_class_to_intent(predicted_class, intent_encoder)
    
    # Generate response
    response = get_response_from_intent(intent)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
