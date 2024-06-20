import re
import stanza
import torch
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords
from string import punctuation
from textblob import TextBlob, Word
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('stopwords')
stanza.download('en')
nlp = stanza.Pipeline('en', processort = 'tokenize')
model_name = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = """
I came here in 2022 and the hotel was excellent! Great room service, restaurant, and extremely warm thermal pool!
I returned now in April 2024 and it doesn't even look like the same place! The pools seem heated, I stayed for 6 days and during those 6 days I questioned the water temperature and no one knew how to answer me, it got to the point that on the last day the water was freezing! Room service stopped a few days ago, towels with holes, sheets with old stains. In the restaurant, the quality of the breakfast frustrated me, in addition to the other meals, the waiters seemed lost and unprepared.
There is a lack of maintenance in the hotel as a whole, a huge loss, as I came here for the thermal water, good food and good rooms, today we don't have any of that anymore.
"""



sentences = []
doc = nlp(text)


def nltk_sentiment(aspect):
    sentiment_aspect = {}
    inputs = tokenizer(text, aspect, return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)

    scores = F.softmax(outputs.logits[0], dim=-1)
    label_id = torch.argmax(scores).item()
    sentiment_aspect[aspect] = (model.config.id2label[label_id], scores[label_id].item())
    return sentiment_aspect





def clean_sentence(sentence):
    sentence = re.sub(r"(?:\@|://)\S+|\n+", "", sentence.lower())
    sent = TextBlob(sentence)
    sent.correct()
    clean = ""
    for sentence in sent.sentences:    
        words = sentence.words
        words = [''.join(c for c in s if c not in punctuation) for s in words]
        words = [s for s in words if s]
        clean += " ".join(words)
        clean += ". "
    return clean


for sentence in doc.sentences:
    sentences.append(' '.join([token.text for token in sentence.tokens]))

result = [clean_sentence(sentence) for sentence in sentences]
comments = TextBlob(' '.join(result))
cleaned = []

# Получаем список всех существительных фраз и удаляем те, которые с ошибками.
for phrase in comments.noun_phrases:
    count = 0
    for word in phrase.split():
        if len(word) <= 2 or (not Word(word).definitions):
            count += 1
    if count < len(phrase.split())*0.4:
        cleaned.append(phrase)


for phrase in cleaned:    
    match = []
    temp = []
    word_match = []
    for word in phrase.split():
        word_match = [p for p in cleaned if re.search(word, p) and p not in word_match]
        if len(word_match) <= len(cleaned)*0.3:
            temp.append(word)
            match += word_match
            
    phrase = ' '.join(temp)

    if len(match) >= len(cleaned)*0.1:
        for feature in match:
            if feature not in cleaned:
                continue
            cleaned.remove(feature)
        cleaned.append(max(match, key=len))
        

feature_count = {}
for phrase in cleaned:
    count = 0
    for word in phrase.split():
        if word not in stopwords.words('english'):
            count += comments.words.count(word)
    feature_count[phrase] = count

counts = list(feature_count.values())
features = list(feature_count.keys())
threshold = len(comments.noun_phrases)/100

frequent_features = []

for feature, count in feature_count.items():
    if count >= threshold:
        frequent_features.append(feature)



nltk_results = [nltk_sentiment(row) for row in frequent_features]

absa_list = {}
for f in frequent_features:
    absa_list[f] = list()
    for comment in result:
        blob = TextBlob(comment)
        for sentence in blob.sentences:
            q = '|'.join(f.split())
            if re.search(r'\w*(' + str(q) + ')\w*', str(sentence)):
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
