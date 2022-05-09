#
# from flask import Flask, render_template
import pandas as pd
import string
import re
import pickle

# app = Flask(__name__)



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
def manual_testing(news,model, vectorization):
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




with open('vector.pkl', 'rb') as file2:
    # Call load method to deserialze
    vectorization = pickle.load(file2)
# Open the file in binary mode
with open('model.pkl', 'rb') as file:
    # Call load method to deserialze
    model = pickle.load(file)

# @app.route('/')
# def home():
#     return render_template('home.html')


news = input("Enter news: ")
manual_testing(news,model,vectorization)

# if __name__ == "__main__":
#     app.run(debug=True)