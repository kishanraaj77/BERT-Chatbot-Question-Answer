from sklearn.externals import joblib
from bert_prediction import *
from flask import request, render_template, Flask

app = Flask(__name__)

qa_model = joblib.load("./models/bert_qa.joblib")
with open('./data/Askari_Bank_FAQ.txt') as f:
    paragraphs = f.readlines()
paragraphs = [x.strip() for x in paragraphs]

def get_response_bert(query, paragraphs):
    json_data = {}
    json_data['version'] = 'v1.1'
    json_data['data'] = list()
    json_data['data'].append({'title': 'Askari Bank QA',
                              'paragraphs': []})

    for para in paragraphs:
        json_data['data'][0]['paragraphs'].append({'context': para, 'qas': []})

    for c, para in enumerate(paragraphs):
        json_data['data'][0]['paragraphs'][c]['qas'].append({"id": c, "question": query})

    processor = BertProcessor(do_lower_case=True, is_training=False)
    example, features = processor.fit_transform(X=json_data['data'])

    bert_return = qa_model.predict(X=(example, features), return_logit=True)
    return bert_return[2]


@app.route("/")
def home():
    return render_template("home.html")
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    #print()
    return get_response_bert(userText, paragraphs)
if __name__ == "__main__":
    app.run()



# In[ ]:




