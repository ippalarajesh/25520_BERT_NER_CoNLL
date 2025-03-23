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