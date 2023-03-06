#imports
import tweepy, config, predict

#setting up a client to make API request to twitter API
def createClient():
    client = tweepy.Client(bearer_token=config.BEARER_TOKEN,
                            consumer_key=config.API_KEY,
                            consumer_secret=config.API_KEY_SECRET,
                            access_token=config.ACCESS_TOKEN,
                            access_token_secret=config.ACCESS_TOKEN_SECRET)
    return client

def predictFromUrl(url,client):
    tweetId=url.split('/')[-1]
    tweet=client.get_tweet(tweetId)
    text=tweet.data.text
    emotion=predict.predictEmotion(text)
    return (emotion,text)

