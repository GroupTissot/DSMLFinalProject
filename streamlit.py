import streamlit as st
import joblib
import numpy as np
import pandas as pd
from transformers import CamembertTokenizer, CamembertModel
import torch
import torch.nn as nn
import tqdm



# Load your trained XGBoost model
csmodel = joblib.load('/Users/mac/Desktop/Python_environment/xgboost.pkl')

# Make sure the device is set correctly for your environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertModel.from_pretrained('camembert-base')
model = model.to(device)

# Function to extract features using BERT
def bert_feature(data, max_length=512):
    # Encode the text, padding to max_length and truncating if necessary
    encoded_input = tokenizer(data, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    encoded_input = encoded_input.to(device)
    
    # Generate embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # We take the first token ([CLS]) embeddings for sentence classification
    features = model_output.last_hidden_state[:, 0, :].cpu().numpy()
    return features

# Function to grade a sentence
def grade_sentence(sentence):
    # Extract features
    test_features = bert_feature([sentence])  # Make sure to pass a list
    print(f"Features shape: {test_features.shape}")  # Debug print statement

    # Predict the grade
    grade = csmodel.predict(test_features)
    print(f"Raw prediction: {grade}")  # Debug print statement

    return grade[0]  # Return the first prediction

# Function to map grades to colors
def get_color(grade):
    # Adjusted to work with actual grade labels
    if grade == "A1" or grade == "A2":
        return "green"
    elif grade == "B1" or grade == "B2":
        return "orange"
    else:
        return "red"


def converter(diff):
      if diff == 0:
        return 'A1'
      elif diff == 1:
        return 'A2'
      elif diff == 2:
        return 'B1'
      elif diff == 3:
        return 'B2'
      elif diff == 4:
        return 'C1'
      else:
        return 'C2'

print(converter(grade_sentence("Nous allons bien, nous habitons dans une petite maison ancienne avec un très beau jardin")))



def main():
    
    st.title("CEFR Sentence Grader")

    sentence = st.text_area("Input your sentence: ")
    st.write(f'You wrote {len(sentence)} characters.')
    
    if sentence:

        grade = grade_sentence(sentence)
        st.subheader("CEFR Grade:")
        st.subheader(converter(grade), divider=get_color(converter(grade)))

    with st.chat_message("user"):
        st.write("Hello 👋")
        st.write("We're group Tissot_UNIL 2023")
        st.write("This application was made for a project in Machine Learning & Data Science.")
        st.write("This uses XGBoost with the parameters below & camembert.")
        st.code('''XGBClassifier(learning_rate=0.2, max_depth=9)''',language = 'python')
        st.write("If you'd like to know more, please visit our github link : ")
        st.link_button("Go to Tissot_UNIL 2023's Github", "https://github.com/GroupTissot/DSMLFinalProject")
        
if __name__ == "__main__":
    main()
