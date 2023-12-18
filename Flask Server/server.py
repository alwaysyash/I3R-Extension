from flask import Flask, request, jsonify,render_template
import json
import base64
import firebase_admin 
from firebase_admin import credentials
from firebase_admin import firestore
import datetime

import pytesseract
from PIL import Image

import spacy
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import numpy as np

import os
from IPython.display import Image
import PIL
import pickle
import clip

import pickle
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
#===============================================================================


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

    # Custom tokenizer function for stop words removal and lemmatization
def custom_tokenizer(text):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

# ===========================================================
# Global Functional dependencies for image model


class ClipWrapper(torch.nn.Module):
    def __init__(self, device, model_name='ViT-L/14'):
        super(ClipWrapper, self).__init__()
        self.clip_model, self.preprocess = clip.load(model_name, device, jit=False)
        self.clip_model.eval()

    def forward(self, x):
        return self.clip_model.encode_image(x)


class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings, device):
        super(SimClassifier, self).__init__()
        self.embeddings = torch.nn.parameter.Parameter(embeddings.to(torch.float))  # Convert to 'Float'


    def forward(self, x):
        embeddings_norm = self.embeddings / self.embeddings.norm(dim=-1,
                                                                 keepdim=True)
        # Pick the top 5 most similar labels for the image
        image_features_norm = x / x.norm(dim=-1, keepdim=True)

        embeddings_norm=embeddings_norm.to(torch.float)
        image_features_norm=image_features_norm.to(torch.float)

        similarity = (100.0 * image_features_norm @ embeddings_norm.T)
        # values, indices = similarity[0].topk(5)
        return similarity.squeeze()


def initialize_prompts(clip_model, text_prompts, device):
    text = clip.tokenize(text_prompts).to(device)
    return clip_model.encode_text(text)


def save_prompts(classifier, save_path):
    prompts = classifier.embeddings.detach().cpu().numpy()
    pickle.dump(prompts, open(save_path, 'wb'))


def load_prompts(file_path, device):
    return torch.HalfTensor(pickle.load(open(file_path, 'rb'))).to(device)

device='cpu'
prompt_path = r'Flask Server\prompts.p'
trained_prompts = load_prompts(prompt_path, device=device)
trained_prompts

clip = ClipWrapper(device)
print('initialized clip model')

classifier = SimClassifier(trained_prompts, device)
print('initialized classifier')

# def compute_embeddings(image_paths):
#     images = clip.preprocess(PIL.Image.open(image_paths))
#     images = torch.stack(images).to(device)
#     return clip(images).half()
def compute_embeddings(image_path):
    image = clip.preprocess(PIL.Image.open(image_path)).unsqueeze(0).to(device)
    return clip(image).half()

#===============================================================================
# Global Functional dependencies for database

# key paste 



cred=credentials.Certificate(key)
default_app = firebase_admin.initialize_app(cred)
print(default_app.name)

db=firestore.client()



#===============================================================================
#===============================================================================
#===============================================================================
# Flask Server


app = Flask(__name__)


@app.route('/submit_report', methods=['GET','POST'])
def submit_report():
    flag="Appropriate"
    # Receive the form data and image from the Chrome extension
    form_data = request.form #can be used for database
    #for local operations
    report_tags=[]
    platforms=[]
    user=[]
    timestamp=datetime.datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d%H%M%S")#for name of image

#===================================================================================================================================================================
#Data Processing
    # Separate the form data from the image
    for key, value in form_data.items():
        if key == 'image':
            # Handle the image separately
            image = value
        elif key.startswith('report_reason['):
            report_tags.append(value)
        elif key=='report_platform':
            platforms.append(value)
        elif key=='custom_platform_value' and value!='':
            platforms.append(value)
        elif key =='user':
            user.append(value)
    
    # Process the form data as needed
    # print("Form Data:")
    # print(json.dumps(form_data, indent=4))
    print("Tags:",report_tags)
    print("Platform:",platforms)
    print("User:",user)

    # Image decode and save
    base64_image_data = image
    base64_image_data = base64_image_data.split(',')[1] # Remove the 'data:image/png;base64,' prefix
    image_data = base64.b64decode(base64_image_data)
    file_path = f'Flask Server/Images/image_{timestamp_str}.png'
    with open(file_path, 'wb') as image_file:
        image_file.write(image_data)

    print("Image saved successfully to", file_path)



#===================================================================================================================================================================
#Image (corpus) Preprocess

    pytesseract.pytesseract.tesseract_cmd = r'Flask Server\Tesseract\tesseract.exe'
    image_path = file_path # Provide the correct file path

    # Open the image using PIL (Python Imaging Library)
    image = PIL.Image.open(image_path)

    # Use Tesseract to extract text from the image
    text = pytesseract.image_to_string(image)

    # Print the extracted text
    print(text)

    nlp = spacy.load("en_core_web_sm")
    ocr_text = ' '.join(text.split())
    doc = nlp(ocr_text)

    # Define a minimum token count for sentences to be considered meaningful
    min_tokens = 5
    # Define a regular expression pattern to identify non-meaningful words
    non_meaningful_pattern = r'^[A-Za-z]*$'
    # Initialize a list to store the meaningful sentences
    meaningful_sentences = []

    for sentence in doc.sents:
        meaningful_tokens = [token.text for token in sentence if not token.is_punct and re.match(non_meaningful_pattern, token.text)]
        if len(meaningful_tokens) >= min_tokens:
            meaningful_sentences.append(" ".join(meaningful_tokens))

    meaningful_text = " ".join(meaningful_sentences)

    # Print the meaningful text
    print(meaningful_text)





# =======================================================================
    # using pickle files for text model

    with open(r'Flask Server\vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    with open(r'Flask Server\random_forest_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    

    new_text = [meaningful_text]
    new_text_vectorized = vectorizer.transform(new_text)
    predicted_label = loaded_model.predict(new_text_vectorized)
    print("Predicted label:", predicted_label[0])
    flag=predicted_label[0]

    
#=====================================================================================================
# Image Model
    image_paths= file_path
    x = compute_embeddings(image_paths)

    y = classifier(x)
    y = torch.argmax(y, dim=0) # label 1 corrosponds to inappropriate material
    # print(y.tolist())

    if y.tolist()==1:
        img_flag= "Inappropriate"
    else:
        img_flag="Appropriate"

    print("Image Flag:",img_flag)

#===================================================================================================================================================================

#Firebase Integration
    data={
        "DateTime": timestamp,
        "Tags":report_tags,
        "Platform": platforms[0],
        "User": user[0],
        "Text Flag":flag,
        "Image Flag":img_flag
    }

    doc_ref = db.collection("I3R")
    doc_ref.add(data)


#===================================================================================================================================================================
# Respond with a JSON message (you can customize this response)
    response = {"message": "Report received and processed"}
    return jsonify(response)


#===================================================================================================================================================================

if __name__ == '__main__':
    app.run(debug=True)





# Global Functional dependencies for text model

#     data = pd.read_csv(r'Flask Server\Model Train\English YouTube Hate Speech Corpus\IMSyPP_EN_YouTube_comments_train.csv')
#     data = data[['Text', 'Type']]
#     data = data[data['Type'] != '0']

#     data['Text'] = data['Text'].str.lower()
#     data=data.dropna()

#     # Split your dataset into training and testing sets
#     X_train_initial = data['Text']


#===================================================================================================================================================================
    #Run image through model

    # # Load model directly
    # tokenizer = AutoTokenizer.from_pretrained("IMSyPP/hate_speech_en")
    # model = AutoModelForSequenceClassification.from_pretrained("IMSyPP/hate_speech_en")

    # # List of text samples you want to classify
    # text_samples = meaningful_text

    # # Tokenize the text samples and get model inputs
    # inputs = tokenizer(text_samples, padding=True, truncation=True, return_tensors="pt")

    # # Forward pass through the model
    # with torch.no_grad():
    #     outputs = model(**inputs)

    # # Get predicted probabilities for each class
    # logits = outputs.logits
    # predicted_probabilities = torch.softmax(logits, dim=1)

    # # Get the predicted class labels (class with the highest probability)
    # predicted_labels = torch.argmax(predicted_probabilities, dim=1)

    # # Define class labels
    # class_labels = ["Acceptable", "Inappropriate", "Offensive", "Violent"]

    # # Print the predictions
    # print(f"Predicted Class: {class_labels[predicted_labels]}")
    # print(f"Predicted Probabilities: {predicted_probabilities}")

    # flag=class_labels[predicted_labels]