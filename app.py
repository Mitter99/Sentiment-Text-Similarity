import numpy as np
#import pandas as pd 
import joblib
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request,render_template
import flask


app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/index')
def index():
   return flask.render_template('index.html')

@app.route('/result',methods = ['POST'])
def result():
   if request.method == 'POST':
      result = request.form.to_dict()
      result = list(result.values())
      print(result[0])
      print(result[1])
      sentences= [result[0],result[1]]
      model = SentenceTransformer('all-MiniLM-L6-v2')
      embeddings = model.encode(sentences)
      cos_sim = cosine_similarity([embeddings[0]],[embeddings[1]])

      return jsonify({'Similarity Score': float(cos_sim)})
    



if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8080)