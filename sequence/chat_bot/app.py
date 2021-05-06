from py2neo import *
from flask import Flask, render_template, request
from sequence.bert_ner.bert_ce_runner import Bert_Runner
from sequence.intention_classification.intention_classification_runner import IntentionClassificationRunner


app = Flask(__name__)
app.static_folder = 'static'

icr = IntentionClassificationRunner('intention_classification_config.yml')
ner = Bert_Runner('bert_ce_config.yml')

class Neo4jImport():
    def __init__(self):
        self.graf = Graph(
            "http://172.22.179.237:7474/",
            user="neo4j",
            password="123456"
        )
        self.email_dict = {}
        self.node_match = NodeMatcher(self.graf)
        self.rel_match = RelationshipMatcher(self.graf)
        pass

    def macth_node(self, keyword, intent):
        sql = 'match (:keyword{name:\'' + keyword + '\'})-[:`' + intent + '`]->(m) return m'
        result = self.graf.run(sql).data()[0]
        return result['m']['name']

neo4j = Neo4jImport()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=['GET'])
def get_bot_response():
    userText = request.args.get('msg')
    keywords = ner.predict_test(userText)[0]
    intent = icr.predict_test(userText)
    result = neo4j.macth_node(keywords, intent)
    return result

if __name__ == "__main__":
    app.run(port=7777, debug=False, host='127.0.0.1')
