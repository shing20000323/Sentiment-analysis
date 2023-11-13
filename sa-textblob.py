import pandas as pd
import seaborn as sns

sns.set(rc={'figure.figsize':(30,1)})

def visualise_sentiments(data):
  sns.heatmap(pd.DataFrame(data).set_index("Sentence").T,center=0, annot=True, cmap = "PiYG")

from textblob import TextBlob
df_reviews = pd.read_csv('reviews-cleaned.csv')
sentence = df_reviews['Description'][9]
TextBlob(sentence).sentiment
visualise_sentiments({
      "Sentence":["SENTENCE"] + sentence.split(),
      "Sentiment":[TextBlob(sentence).polarity] + [TextBlob(word).polarity for word in sentence.split()],
      "Subjectivity":[TextBlob(sentence).subjectivity] + [TextBlob(word).subjectivity for word in sentence.split()],
})

print(df_reviews)

#plotting the sentiment of the review

import matplotlib.pyplot as plt

df_reviews['Sentiment'] = df_reviews['Description'].apply(lambda x: TextBlob(x).polarity)
df_reviews['Subjectivity'] = df_reviews['Description'].apply(lambda x: TextBlob(x).subjectivity)
plt.show()
