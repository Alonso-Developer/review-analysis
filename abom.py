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

text = """
I have never experienced such dissatisfaction with the lack of empathy from a person responsible for a reception at a hotel of the magnitude of Wyndham where I was accommodated in a room that did not match the reservation made without any ability to adequately accommodate the number of people described in the reservation. I was sold one thing and offered another and still had problems and at no point was the hotel willing to resolve or try to resolve my problem, on the contrary the attendant was harsh with me and so I could try to resolve it myself, I made another reservation to accommodate everyone properly. And amazingly, I asked to at least move my check-in forward a few hours and was denied by the same employee responsible for the reception. Zero empathy and lack of care for a satisfied customer Terrible experience.
"""


doc = nlp(text)

def nltk_sentiment(text, aspect):
    sentiment_aspect = {}
    inputs = tokenizer(text, aspect, return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)

    scores = F.softmax(outputs.logits[0], dim=-1)
    label_id = torch.argmax(scores).item()
    sentiment_aspect[aspect] = (model.config.id2label[label_id], scores[label_id].item())
    return sentiment_aspect



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
    absa_list[f] = list()
    for comment in comments.sentences:
        blob = TextBlob(str(comment))
        for sentence in blob.sentences:
            if re.search(str(f).lower(), str(sentence).lower()):
                absa_list[f].append(sentence)


print("[ REVIEW ]")
print(text)

for key in nltk_results:
    dict_key = list(key.keys())[0]
    dict_res = key[dict_key]
    print(dict_key, dict_res[0], dict_res[1])
print("\nAspect Specific sentences:\n")

for i in absa_list:
    print(i)
    for j in absa_list[i]:
        print(f"\t- {str(j)}")
    print()
