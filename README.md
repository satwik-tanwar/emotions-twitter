# emotions-twitter
This project classifies the emotion of a tweet. You can either copy paste the tweet in the form provided in the webapp or just paste the link of the tweet.It uses twitter api to extract the tweet and then predicts the following emotions using a BERT-based emotion classifier - 
1. Anger 
2. Love 
3. Fear
4. Joy
5. Sadness
6. Surprise

## Running the project
1. Clone the repository
2. Create a new .env file and add the following in it - 
```
API_KEY = your_twitter_api_key
API_KEY_SECRET = your_twitter_api_key_secret
BEARER_TOKEN = your_twitter_api_bearer_token
ACCESS_TOKEN = your_twitter_api_access_token
ACCESS_TOKEN_SECRET = your_twitter_api_access_token_secret
```
3. Install the requirements in your environment
```bash
$ pip install -r requirements.txt
```
4. Run app.py
```bash
$ python app.py
```