from flask import Flask, render_template, url_for, redirect, request
from predict import predictEmotion
import config, search

def page_not_found(e):
  return render_template('404.html'), 404


app = Flask(__name__)
#app.config.from_object(config.config['development'])   #for development
app.config.from_object(config.config['production'])     #for production

app.register_error_handler(404, page_not_found)



@app.route('/', methods=["GET"])
def index():
    data={"textLabel":"","text":"","emotion":""}     
    return render_template('index.html', data=data)

@app.route('/tweet',methods=["POST"])
def predTweet():
    data={"textLabel":"Tweet","text":"","emotion":""}      
    sentence=request.form['sentence']
    emotion=predictEmotion(sentence)
    data['text']=sentence
    data['emotion']=emotion
    return render_template('index.html', data=data)
   
@app.route('/tweetUrl',methods=["POST"])
def predTweetFromURL():
    client=search.createClient()
    data={"textLabel":"Tweet","text":"","emotion":""}   
    url=request.form['url']
    emotion,text=search.predictFromUrl(url,client)
    data['text']=text
    data['emotion']=emotion
    return render_template('index.html', data=data)   

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port='8000', debug=True)
