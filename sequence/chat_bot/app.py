# from chatbot import chatbot
from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=['GET'])
def get_bot_response():
    userText = request.args.get('msg')
    return userText

if __name__ == "__main__":
    app.run(port=7777, debug=False, host='127.0.0.1')
