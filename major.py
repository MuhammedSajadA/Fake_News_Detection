#
from flask import Flask, render_template, request, jsonify
import pandas as pd
import string
import re
import pickle
from flask_cors import CORS

app = Flask(__name__)
cors=CORS()
cors.init_app(app)


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"


from sklearn.feature_extraction.text import TfidfVectorizer


def manual_testing(news, model, vectorization):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]

    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = model.predict(new_xv_test)
    #     #     pred_DT = DT.predict(new_xv_test)
    #     #     pred_SVM = SVM.predict(new_xv_test)
    #     #     pred_RFC = RFC.predict(new_xv_test)
    #     #     pred_KNN = clf2.predict(new_xv_test)
    print(output_lable(pred_LR[0]))
    return output_lable(pred_LR[0])

    print("\n\nLR Prediction: {}  ".format(output_lable(pred_LR[0])))


#                          #                          output_lable(pred_DT[0]),
#                          #                          output_lable(pred_SVM[0]),
#                          #                          output_lable(pred_RFC[0]),
#                          #                          output_lable(pred_KNN[0]),

#
#
# # In[37]:
#
# news = input()
# val = manual_testing(news)
# print(val)


with open('Temp/vector.pkl', 'rb') as file2:
    # Call load method to deserialze
    vectorization = pickle.load(file2)
# Open the file in binary mode
with open('Temp/model.pkl', 'rb') as file:
    # Call load method to deserialze
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/data/', methods=['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form.get('input')
        news = str(form_data)
        ans = manual_testing(news, model, vectorization)
        return render_template('data.html',form_data =ans)

# @app.route('/react')
# def react1():
#     if request.method == 'GET':
#         news=request.json["data"]
#         ans = manual_testing(news, model, vectorization)
#         return jsonify(ans)

@app.route('/react',methods=['POST', 'GET'])
def react1():
        news = request.json["data"]
        ans = manual_testing(news, model, vectorization)
        print(ans)
        print(news)
        return jsonify(ans)

if __name__ == "__main__":
    app.run(debug=True)
