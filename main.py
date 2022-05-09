#!/usr/bin/env python
# coding: utf-8
from flask import Flask, render_template, request

# In[4]:
print("Hello")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string

app = Flask(__name__)




print("TEST")

df_fake = pd.read_csv("Temp\Fake.csv")
df_true = pd.read_csv("Temp\True.csv")

# In[6]:


df_fake.head(10)

# In[7]:


df_true.head(10)

# In[8]:


df_fake['class'] = 0
df_true['class'] = 1

# In[9]:


df_fake.shape, df_true.shape

# In[10]:


df_fake_manual_testing = df_fake.tail(10)
for i in range(23470, 23460, -1):
    df_fake.drop([i], axis=0, inplace=True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416, 21406, -1):
    df_true.drop([i], axis=0, inplace=True)

# In[18]:


df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1

# In[11]:


df_fake_manual_testing.head(10)

# In[12]:


df_true_manual_testing.head(10)

# In[13]:


df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
df_manual_testing.to_csv("Temp\manual_testing.csv")

# In[14]:


df_marge = pd.concat([df_fake, df_true], axis=0)
df_marge.head(10)

# In[15]:


df = df_marge.drop(["title", "subject", "date"], axis=1)

# In[16]:


df.head(10)

# In[17]:


df = df.sample(frac=1)

# In[18]:


df.head(10)

# In[19]:


df.isnull().sum()


# In[20]:


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


# In[22]:


df["text"] = df["text"].apply(wordopt)

# In[23]:


df.head(10)

# In[24]:


x = df["text"]
y = df["class"]

# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer

# In[27]:


vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# ### 1. Logistic Regression

# In[28]:


from sklearn.linear_model import LogisticRegression

# In[29]:


LR = LogisticRegression()
LR.fit(xv_train, y_train)

# In[30]:


pred_lr = LR.predict(xv_test)

# In[31]:


LR.score(xv_test, y_test)

# In[32]:


print(classification_report(y_test, pred_lr))


# # ### 2. Decision Tree Classification
#
# # In[40]:
#
#
# from sklearn.tree import DecisionTreeClassifier
#
#
# # In[41]:
#
#
# DT = DecisionTreeClassifier()
# DT.fit(xv_train, y_train)
#
#
# # In[42]:
#
#
# pred_dt = DT.predict(xv_test)
#
#
# # In[43]:
#
#
# DT.score(xv_test, y_test)
#
#
# # In[44]:
#
#
# print(classification_report(y_test, pred_dt))
#
#
# # ### 3. Random Forest
#
# # In[45]:
#
#
# from sklearn.ensemble import RandomForestClassifier
#
#
# # In[46]:
#
#
# RFC = RandomForestClassifier(random_state=0)
# RFC.fit(xv_train, y_train)
#
#
# # In[47]:
#
#
# pred_rfc = RFC.predict(xv_test)
#
#
# # In[48]:
#
#
# RFC.score(xv_test, y_test)
#
#
# # In[49]:
#
#
# print(classification_report(y_test, pred_rfc))
#
#
# # #### 4. The k-Nearest Neighbors
#
# # In[50]:
#
#
# from sklearn.neighbors import KNeighborsClassifier
#
#
# # In[51]:
#
#
# clf2 = KNeighborsClassifier(n_neighbors=9)
# clf2.fit(xv_train, y_train)
#
#
# # In[52]:
#
#
# pred_clf = clf2.predict(xv_test)
#
#
# # In[53]:
#
#
# clf2.score(xv_test, y_test)
#
#
# # In[43]:
#
#
# print(classification_report(y_test, pred_clf))
#
#
# # ### 5. Support Vector Machine
#
# # In[54]:
#
#
# from sklearn import svm
#
#
# # In[ ]:
#
#
# SVM = svm.SVC()
# SVM.fit(xv_train, y_train)
#
#
# # In[46]:
#
#
# pred_svm = SVM.predict(xv_test)
#
#
# # In[47]:
#
#
# SVM.score(xv_test, y_test)
#
#
# # In[48]:
#
#
# print(classification_report(y_test, pred_svm))
#

# # Model Testing With Manual Entry
# 
# ### News

# In[36]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"


def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    #     pred_DT = DT.predict(new_xv_test)
    #     pred_SVM = SVM.predict(new_xv_test)
    #     pred_RFC = RFC.predict(new_xv_test)
    #     pred_KNN = clf2.predict(new_xv_test)

    return print("\n\nLR Prediction: {}  "
                 .format(output_lable(pred_LR[0]),
                         #                          output_lable(pred_DT[0]),
                         #                          output_lable(pred_SVM[0]),
                         #                          output_lable(pred_RFC[0]),
                         #                          output_lable(pred_KNN[0]),
                         ))


# In[37]:

@app.route('/')
def home():
    return render_template('home.html')

# In[5]:


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['a']
        val = manual_testing(news)
        print(val)
    return render_template('out.html')

if __name__ == "__main__":
    app.run(debug=True)
# In[39]:


# In[40]:


# In[ ]:
