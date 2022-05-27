import torch as torch
import numpy as np
import matplotlib.pyplot as plt
from model import *
from gen_data import *
import os
import seaborn as sns
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import *
import spacy
from spacy import displacy
from pathlib import Path

#Initialize Flask and set the template folder to "template"
# app = Flask(__name__, template_folder = 'template')
# app = Flask(__name__)
# app.config.from_object(__name__)
# enable CORS
# cors = CORS(app, resources={r'/*': {'origins': '*'}})
# cors = CORS(app, supports_credentials=True)
# app.config['CORS_HEADERS'] = 'Content-Type'
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin') or 'http://127.0.0.1:5000'
    response.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,Accept,Origin,Referer,User-Agent'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response


app = Flask(__name__)
app.after_request(after_request)

# preparation
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print('Using GPU')
# else:
device = torch.device('cpu')

id2label,label2id = get_string2id('/home/xss/NER/BiLSTM/data3/dataset/CN/labels.txt')
id2vocab,vocab2id = get_string2id('/home/xss/NER/BiLSTM/data3/dataset/CN/vocab.txt')
max_len = 512
model = LSTM_CRF(vocab2id).to(device)
model.load_state_dict(torch.load('/home/xss/NER/BiLSTM/A-web/ckpt/CN-best_model_f90.pth').cpu().state_dict())
model.eval()

#create our "home" route using the "website.html" page
@app.route('/')
def home():
    # return send_from_directory('template', 'website.html')
    return 'hello'

# render_template('website.html')

#Set a post method to yield predictions on page
@app.route('/predict', methods = ['GET', 'POST'])#, methods = ['GET','POST']
def predict():
    # print(request.args)
    # print(request.args.get('sentence'))
    # print(request.args.get('sentence').strip())
    #Get the sentence from the user

    sentence = request.args.get('sentence')
    # sentence = sentence.strip()
    # sentence1 = [vocab2id[word] for word in sentence]
    #
    # sentence_pad = sentence1 + [len(vocab2id)]*(max_len-len(sentence1))
    # sentence_pad = torch.LongTensor(sentence_pad).to(device)
    # sentence_pad = sentence_pad.unsqueeze(0)
    # sentence_pad = torch.cat([sentence_pad]*128, dim=0)
    # with torch.no_grad():
    #     mask = torch.logical_not(torch.eq(sentence_pad, torch.tensor(4465)))
    #     data_y = torch.zeros((128, max_len), dtype=torch.long).to(device)
    #     targets_pred_without_pad, crf_loss, _ = model(sentence_pad, data_y, mask)
    #     # print(targets_pred_without_pad[0])
    #     tags = [id2label[i] for i in targets_pred_without_pad[0]]
    #     print("真实句子", sentence)  # ''.join(juzi)
    #     # print("预测标签", tags)
    #     return jsonify(str(tags))
    # return jsonify({"result": "hello"})

    nlp = spacy.load("zh_core_web_sm")
    doc = nlp(sentence)
    ent = displacy.render(doc, style="ent")
    options = {"compact": True, "bg": "#E4EBF500",
               "color": "#6d5dfc70", "font": "Source Sans Pro"}
    dep = displacy.render(doc, style="dep",options=options)
    filepath = './static/' + 'ent.svg'
    file_name = Path(filepath)
    file_name.open('w', encoding='utf-8').write(dep)
    return {'ent': ent, 'dep': dep}




if __name__ == "__main__":
        app.run(debug=True)