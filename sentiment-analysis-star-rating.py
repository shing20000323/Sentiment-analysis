from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")

model = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis")

      
#predict the sentiment of the review in reviews.csv
import pandas as pd
df_reviews = pd.read_csv('reviews-cleaned.csv')

df_reviews['Sentiment'] = df_reviews['Description'].apply(lambda x: tokenizer(x, return_tensors="pt"))
df_reviews['Sentiment'] = df_reviews['Sentiment'].apply(lambda x: model(**x)[0].argmax().item())
df_reviews.to_csv('reviews-sentiment-star-rating.csv', index=False)
print(df_reviews)
df_reviews.groupby('Sentiment').count()['Name'].plot(kind='bar')
plt.title("Star Count")
plt.show()