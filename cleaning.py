import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

df_reviews = pd.read_csv('reviews.csv')

#Remove numbers, Stemming/Lemmatization, Part of speech tagging, Remove punctuation, Lowercase, Remove stopwords
def clean_text(text):
    text = ''.join([word for word in text if not word.isdigit()])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([WordNetLemmatizer().lemmatize(word) for word in text.split()])
    return text

df_reviews['Description'] = df_reviews['Description'].apply(lambda x: clean_text(x))
df_reviews.to_csv('reviews-cleaned.csv', index=False)
print(df_reviews)
