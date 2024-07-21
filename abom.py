import re
import stanza
import csv
import pandas as pd
import torch
import torch.nn.functional as F
from textblob import TextBlob, Word
from transformers import AutoTokenizer, AutoModelForSequenceClassification

stanza.download('en')
nlp = stanza.Pipeline('en', processort = 'tokenize')
model_name = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
dataset = pd.read_csv('./London_reviews.csv')


def nltk_sentiment(text, aspect):
    sentiment_aspect = {}
    inputs = tokenizer(text, aspect, return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)

    scores = F.softmax(outputs.logits[0], dim=-1)
    label_id = torch.argmax(scores).item()
    sentiment_aspect[aspect] = (model.config.id2label[label_id], scores[label_id].item())
    return sentiment_aspect



def process_review(hotel_name, review):
    text = review
    doc = nlp(text)

    comments = TextBlob(text)
    cleaned = []

    for phrase in comments.noun_phrases:
        count = 0
        for word in phrase.split():
            if len(word) <= 2 or (not Word(word).definitions):
                count += 1
        if count < len(phrase.split())*0.4:
            cleaned.append(phrase)



    nltk_results = [nltk_sentiment(text, row) for row in cleaned]

    absa_list = {}
    for f in cleaned:
        absa_list[f] = []
        sentences = [comment for comment in comments.sentences if re.search(str(f).lower(), str(comment).lower())]
        absa_list[f].extend(sentences)

    try:
        with open(f'./results/{hotel_name}_semantic.txt', 'a') as file:
            for key in nltk_results:
                dict_key = list(key.keys())[0]
                dict_res = key[dict_key]
                file.write(f"{dict_key} {dict_res[0]} {dict_res[1]}\n")
        
        with open(f'./results/{hotel_name}.txt', 'a') as file:
            for i in absa_list:
                file.write(i + '\n')
                for j in absa_list[i]:
                    file.write(f"\t- {str(j)}\n")
                file.write('\n')
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            exit()
        print(e)

dataset.apply(lambda x: process_review(x['restaurant_name'], x['review_full']), axis = 1)

