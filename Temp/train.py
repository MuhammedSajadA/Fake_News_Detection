import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string
import pickle

df_fake = pd.read_csv("Temp/Fake.csv")
df_true = pd.read_csv("Temp/True.csv")

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
df_manual_testing.to_csv("manual_testing.csv")

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


# from sklearn.linear_model import LogisticRegression

# In[29]:


# LR = LogisticRegression()
# LR.fit(xv_train, y_train)

# In[30]:


# pred_lr = LR.predict(xv_test)

# In[31]:


# LR.score(xv_test, y_test)


# print(classification_report(y_test, pred_lr))


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
from sklearn import svm
#
#
# # In[ ]:
#
#
SVM = svm.SVC()
SVM.fit(xv_train, y_train)
#
#
# # In[46]:
#
#
pred_svm = SVM.predict(xv_test)
#
#
# # In[47]:
#
#
SVM.score(xv_test, y_test)
#
#
# # In[48]:
#
#
print(classification_report(y_test, pred_svm))
#

# # Model Testing With Manual Entry
#
# ### News

# In[36]:

with open('vector_svm.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(vectorization, file)
# In[32]:
with open('model_svm.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(SVM, file)

print(classification_report(y_test, pred_svm))


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
