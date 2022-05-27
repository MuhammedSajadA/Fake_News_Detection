# import CORS as/ CORS
from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource
import pickle
import pandas as pd
# from flask_cors import CROS
from sklearn.linear_model import LogisticRegression
# from .models import NLPModel

app = Flask(__name__)

# CORS(app)

api = Api(app)
model = LogisticRegression()

with open('Temp/vector.pkl', 'rb') as file2:
    # Call load method to deserialze
    model.clf = pickle.load(file2)
# Open the file in binary mode
with open('Temp/model.pkl', 'rb') as file:
    # Call load method to deserialze
    model.vectorizer = pickle.load(file)

parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        # vectorize the user's query and make a prediction
        uq_vectorized = model.vectorizer_transform(np.array([user_query]))
        prediction = model.predict(uq_vectorized)
        pred_proba = model.predict_proba(uq_vectorized)

        # Output either 'Negative' or 'Positive' along with the score
        if prediction == 0:
            pred_text = 'Negative'
        else:
            pred_text = 'Positive'

        # round the predict proba value and set to new variable
        confidence = round(pred_proba[0], 3)

        output = {'prediction': pred_text, 'confidence': confidence}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')

if __name__ == '__main__':
    app.run(debug=True)