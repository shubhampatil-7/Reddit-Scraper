
import praw
import pandas as pd
import datetime as dt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

reddit = praw.Reddit(
    client_id = <client_id>, \
    client_secret = <client_secret>, \
    user_agent = <user_agent>, \
    username = <username>, \
    password = <password>
)

headlines = set()
for submission in reddit.subreddit('politics').top(limit=20):
    headlines.add(submission.title)
   

df= pd.DataFrame(headlines)
df.head()
df.to_csv("headlines.csv", header=False, encoding='utf-8', index=False)

# SENTIMENT ANALYSIS
nltk.download('vader_lexicon')

sia = SIA()
results = []

for line in headlines:
    pol_score = sia.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)

print(results[:3])

df = pd.DataFrame.from_records(results)
df.head()
df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1

df2 = df[['headline', 'label']]
df.to_csv('reddit_headlines_labels.csv', encoding='utf-8', index=False)


print("-----------------------------------------------")
print(df.label.value_counts())