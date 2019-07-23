import sys
sys.path.append('../')

from models.train_classifier import *
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from joblib import dump, load
from sqlalchemy import create_engine
import cloudpickle as cp
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import spacy 
import scattertext as st

app = Flask(__name__)

def get_image(ax):
    fig = ax.get_figure()
    buf = BytesIO()
    fig.savefig(buf, format='png',bbox_inches='tight')
    buf.seek(0)
    encoded_file = base64.b64encode(buf.read()).decode('ascii')
    return(encoded_file)
    
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Main', engine)

# load model
#random_state=1234
#np.random_state=random_state 
#estimator=ExtraTreeClassifier(random_state=random_state,class_weight='balanced')
#model = build_model(estimator)

with open("../models/classifier.pkl", 'rb') as f:
    model=cp.load(f)
nlp = spacy.load("en_vectors_web_lg")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    to_predict=pd.DataFrame()
    to_predict['message']=np.array([query])
    vec,_,tokens,_=doc2vec(to_predict.message.values,[0],nlp)
    predicted_a,predicted_p = model.oracle(vec)
    res_df = pd.DataFrame() 
    res_df['cat.']=model.category_names
    res_df['prb.']=predicted_p[0]
    res_df['abs.']=predicted_a[0]
    res_df=res_df.sort_values('prb.',ascending=True)
    ax_1=res_df.plot.barh(x='cat.',y='prb.',
                           figsize=(12,6),legend=False,fontsize=8)
    threshold=np.mean(res_df['prb.'].values)+np.mean(res_df['prb.'].values)/5
    p_threshold=res_df.\
            loc[res_df['abs.']==1].\
            sort_values('prb.',ascending=False).\
            iloc[-1:]['prb.'].values.item()
    ax_1.axvline(x=threshold,color='red')
    print(p_threshold)
    ax_1.axvline(x=p_threshold,color='blue')
    ax_2=res_df.plot.bar(x='cat.',y='abs.',figsize=(15,10),fontsize=.5)
    prb_image=get_image(ax_1)
    #.loc[res_df['prb.']>=threshold]
    res_table=res_df.\
            sort_values('prb.',ascending=False).\
            reset_index(drop=True).\
            to_html(classes='table table-striped')
    
    return render_template(
        'go.html',
        query=query,prb_image=prb_image,res_table=res_table,tokens=tokens[0]
    )#classification_result=classification_results


def main():

    app.run(debug=False)


if __name__ == '__main__':
    main()
