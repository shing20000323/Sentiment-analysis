import pandas as pd
import seaborn as sns

sns.set(rc={'figure.figsize':(30,1)})

def visualise_sentiments(data):
  sns.heatmap(pd.DataFrame(data).set_index("Sentence").T,center=0, annot=True, cmap = "PiYG")
df_reviews = pd.read_csv('reviews.csv')
sentence = df_reviews['Description'][9]
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
sid.polarity_scores(sentence)
visualise_sentiments({
    "Sentence":["SENTENCE"] + sentence.split(),
    "Sentiment":[sid.polarity_scores(sentence)["compound"]] + [sid.polarity_scores(word)["compound"] for word in sentence.split()]
})

#predict the sentiment of the review in reviews.csv
import pandas as pd

import matplotlib.pyplot as plt
df_reviews = pd.read_csv('reviews-cleaned.csv')
df_reviews['Sentiment'] = df_reviews['Description'].apply(lambda x: sid.polarity_scores(x)["compound"])

print(df_reviews)
#plotting the sentiment of the review

df_reviews['Sentiment'] = df_reviews['Description'].apply(lambda x: sid.polarity_scores(x)["compound"])
df_reviews['Subjectivity'] = df_reviews['Description'].apply(lambda x: sid.polarity_scores(x)["pos"])
plt.show()
#show sid.polarity_scores(sentence)
print(sid.polarity_scores(sentence))
df_reviews.to_csv('reviews-sentiment-nltk.csv', index=False)