import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

df_reviews = pd.read_csv('reviews.csv')

#using regex to clean the description, removing usernames, urls, stopwords, etc

df_reviews['reviews-cleaned'] = df_reviews['Description'].str.replace('@[^\s]+', '')
df_reviews['reviews-cleaned'] = df_reviews['Description'].str.replace('[^\w\s]','')
df_reviews['reviews-cleaned'] = df_reviews['Description'].str.replace('\d+', '')
df_reviews['reviews-cleaned'] = df_reviews['Description'].str.replace('\n', '')
df_reviews['reviews-cleaned'] = df_reviews['Description'].str.lower()
df_reviews['reviews-cleaned'] = df_reviews['Description'].str.split()
stop_words = set(stopwords.words('english'))
df_reviews['reviews-cleaned'] = df_reviews['reviews-cleaned'].apply(lambda x: [item for item in x if item not in stop_words])
lemmatizer = WordNetLemmatizer()
df_reviews['reviews-cleaned'] = df_reviews['reviews-cleaned'].apply(lambda x: [lemmatizer.lemmatize(item) for item in x])
df_reviews['reviews-cleaned'] = df_reviews['reviews-cleaned'].apply(lambda x: ' '.join(x))
df_reviews.to_csv('reviews-cleaned.csv', index=False)
print(df_reviews)