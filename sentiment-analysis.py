from transformers import pipeline
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

classifier = pipeline("sentiment-analysis")
df_reviews = pd.read_csv('reviews-cleaned.csv')
#classifier the sentiment of the review 
df_reviews['Sentiment'] = df_reviews['reviews-cleaned'].apply(lambda x: classifier(x)[0]['label'])
df_reviews['Score'] = df_reviews['reviews-cleaned'].apply(lambda x: classifier(x)[0]['score'])
df_reviews.to_csv('reviews-sentiment.csv', index=False)
#show the reviews-sentiment.csv
print(df_reviews)
#Data Visualization
df_reviews = pd.read_csv('reviews-sentiment.csv')
df_reviews.groupby('Sentiment').count()['Name'].plot(kind='bar')
plt.title("Sentiment Value Count")
plt.show()