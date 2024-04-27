from flask import Flask, render_template
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
import plotly.graph_objects as go

app = Flask(__name__)

trump_reviews = pd.read_csv("Trumpall2.csv")
biden_reviews = pd.read_csv("Bidenall2.csv")


textblob1 = TextBlob(trump_reviews["text"][2000])
print("Trump :",textblob1.sentiment)
textblob2 = TextBlob(biden_reviews["text"][2000])
print("Biden :",textblob2.sentiment)



def find_pol(review):
    return TextBlob(review).sentiment.polarity


#       ****   tail reviews     ****

trump_reviews["Sentiment Polarity"] = trump_reviews["text"].apply(find_pol)
trump_tail= trump_reviews.tail()
# print(trump_reviews.tail())



biden_reviews["Sentiment Polarity"] = biden_reviews["text"].apply(find_pol)
biden_tail= biden_reviews.tail()
# print(biden_reviews.tail())


#       ****    tail reviews with polarity  ****
trump_reviews["Expression Label"] = np.where(trump_reviews["Sentiment Polarity"]>0, "positive", "negative")
trump_reviews["Expression Label"][trump_reviews["Sentiment Polarity"]==0]="Neutral"

trump_tail_polarity = trump_reviews.tail()

biden_reviews["Expression Label"] = np.where(biden_reviews["Sentiment Polarity"]>0, "positive", "negative")
biden_reviews["Expression Label"][trump_reviews["Sentiment Polarity"]==0]="Neutral"
biden_tail_polarity = biden_reviews.tail()



#       **** Anonymous  ****
reviews1 = trump_reviews[trump_reviews['Sentiment Polarity'] == 0.0000]
print(reviews1.shape)

cond1=trump_reviews['Sentiment Polarity'].isin(reviews1['Sentiment Polarity'])
trump_reviews.drop(trump_reviews[cond1].index, inplace = True)
print(trump_reviews.shape)

reviews2 = biden_reviews[biden_reviews['Sentiment Polarity'] == 0.0000]
print(reviews2.shape)

cond2=biden_reviews['Sentiment Polarity'].isin(reviews1['Sentiment Polarity'])
biden_reviews.drop(biden_reviews[cond2].index, inplace = True)
print(biden_reviews.shape)


#       ****    Ploting Graph   ****
# Donald Trump
np.random.seed(10)
remove_n =324
drop_indices = np.random.choice(trump_reviews.index, remove_n, replace=False)
df_subset_trump = trump_reviews.drop(drop_indices)
print(df_subset_trump.shape)


# Joe Biden
np.random.seed(10)
remove_n =31
drop_indices = np.random.choice(biden_reviews.index, remove_n, replace=False)
df_subset_biden = biden_reviews.drop(drop_indices)
print(df_subset_biden.shape)


count_1 = df_subset_trump.groupby('Expression Label').count()
print(count_1)

negative_per1 = (count_1['Sentiment Polarity'][0]/1000)*10
positive_per1 = (count_1['Sentiment Polarity'][1]/1000)*100

count_2 = df_subset_biden.groupby('Expression Label').count()
print(count_2)

negative_per2 = (count_2['Sentiment Polarity'][0]/1000)*100
positive_per2 = (count_2['Sentiment Polarity'][1]/1000)*100

Politicians = ['Joe Biden', 'Donald Trump']
lis_pos = [positive_per1, positive_per2]
lis_neg = [negative_per1, negative_per2]

fig = go.Figure(data=[
    go.Bar(name='Positive', x=Politicians, y=lis_pos),
    go.Bar(name='Negative', x=Politicians, y=lis_neg)
])

fig.update_layout(barmode='group')



trump_reviews_sorted = trump_reviews.sort_values(by="Sentiment Polarity", ascending=False)
biden_reviews_sorted = biden_reviews.sort_values(by="Sentiment Polarity", ascending=False)



top5_trump_positive = trump_reviews_sorted.head(5)
top5_trump_negative = trump_reviews_sorted.tail(5)

top5_biden_positive = biden_reviews_sorted.head(5)
top5_biden_negative = biden_reviews_sorted.tail(5)


NEWS_API_KEY = '4c2fd060b4544b49ae94cd0159525ba3'

def get_news(topic):
    url = f'https://newsapi.org/v2/everything?q={topic}&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get('articles', [])
        return articles
    else:
        return None


@app.route('/')
def index():
    
    
    top5_trump_negative_reviews = top5_trump_negative['text']
    top5_biden_negative_reviews = top5_biden_negative['text']
    
    biden_articles = get_news('Joe Biden')
    trump_articles = get_news('Donald Trump')
    
    all_articles = biden_articles + trump_articles
    
    return render_template('index.html', all_articles=all_articles,plot_div=fig.to_html(full_html=False),trump_reviews=top5_trump_negative_reviews, biden_reviews=top5_biden_negative_reviews)


@app.route('/sentiment')
@app.route('/sentiment')
def sentiment():
    top5_trump_positive_reviews =   top5_trump_positive['text']
    top5_biden_positive_reviews =   top5_biden_positive['text']
    top5_trump_negative_reviews = top5_trump_negative['text']
    top5_biden_negative_reviews = top5_biden_negative['text']
    return render_template('sentiment.html', trump_reviews=top5_trump_negative_reviews, biden_reviews=top5_biden_negative_reviews,trump_reviews_positive=top5_trump_positive_reviews,biden_reviews_positive=top5_biden_positive_reviews)

@app.route('/visualization')
def visualization():
    return render_template('visualization.html', plot_div=fig.to_html(full_html=False))

if __name__ == '__main__':
    app.run(debug=True)
