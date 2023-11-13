import pandas as pd
import seaborn as sns

sns.set(rc={'figure.figsize':(30,1)})

def visualise_sentiments(data):
  sns.heatmap(pd.DataFrame(data).set_index("Sentence").T,center=0, annot=True, cmap = "PiYG")

from textblob import TextBlob
df_reviews = pd.read_csv('reviews.csv')
sentence = df_reviews['Description'][9]
TextBlob(sentence).sentiment
visualise_sentiments({
      "Sentence":["SENTENCE"] + sentence.split(),
      "Sentiment":[TextBlob(sentence).polarity] + [TextBlob(word).polarity for word in sentence.split()],
      "Subjectivity":[TextBlob(sentence).subjectivity] + [TextBlob(word).subjectivity for word in sentence.split()],
})
#plotting the sentiment of the review




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(30,1)})

df_reviews['Sentiment'] = df_reviews['Description'].apply(lambda x: TextBlob(x).polarity)
df_reviews['Subjectivity'] = df_reviews['Description'].apply(lambda x: TextBlob(x).subjectivity)
plt.title("Subjectivity Value Count")
plt.show()
