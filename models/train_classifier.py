import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import spacy
import en_vectors_web_lg
from pycm import *
from langdetect import detect
from langdetect import detect_langs
from spellchecker import SpellChecker
import cloudpickle as cp

from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

#following should be installed
#'https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz'

def load_data(database_filepath="data/DisasterResponse.db",table_name="Main"):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name, engine)
    #a new feature named 'unknown' is added to detect records with all features set to 0 
    df['unknown']=df.iloc[:,4:].apply(lambda x:int(len(np.unique(x))==1 & all(np.unique(x)==[0])),axis=1)
    #No examples available for child_alone to train the model, this feature is removed
    df.drop(['child_alone'],inplace=True,axis=1)
    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names= df.iloc[:,4:].columns
    return(X,Y,category_names)

def clean_text(target_docs):
    nlp = spacy.load("en_vectors_web_lg")
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    output=[]
    delete_index=[]
    print('Cleaning Processing Started...')
    for i,input_string in enumerate(list(target_docs)):
        string=str.lower(input_string)
        doc = nlp(string)
        lemmatized=[]
        for token in doc:
            if((not token.is_punct) & (token.text not in spacy_stopwords)& (token.has_vector)):
                lemmatized.append(token.lemma_.strip())
        new_string=" ".join(lemmatized)
        output.append(new_string)
    print('Cleaning Done...')
    return(output)

def count2vec(target_docs):
    cleaned_docs=clean_text(target_docs)
    CV=CountVectorizer()
    print('CountVectorizer Started...')
    results=CV.fit_transform(cleaned_docs)
    print('CountVectorizer Finished...')
    return(results.toarray())

def tfidf2vec(target_docs,**kwargs):
    cleaned_docs=clean_text(target_docs)
    TF_IDF=TfidfVectorizer(**kwargs)
    print('TfidfVectorizer Started...')
    results=TF_IDF.fit_transform(cleaned_docs)
    print('TfidfVectorizer Finished...')
    return(results.toarray())

def spell_check(target,nlp):
    from spellchecker import SpellChecker
    spell_checked=[]
    spell = SpellChecker()
    doc=nlp(target)
    for token in doc:
        new_token=''
        if(token.text in nlp.vocab):
            new_token=token.text
        elif(token.text!=spell.correction(token.text)):
            new_token=spell.correction(token.text)
        if(new_token):
            spell_checked.append(new_token)
    new_doc=nlp(" ".join(spell_checked))
    lemmatized=[token.lemma_.strip() for token in new_doc]
    return(" ".join(lemmatized))
def remove_punct(target):
    import string
    output=target.translate(str.maketrans(string.punctuation,
                 ' ' * len(string.punctuation))).replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ').strip()
    return(output)

def doc2vec(target_docs,target_categories,nlp):
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    output=[]
    delete_index=[]
    cleaned_docs=[]
    all_removed_tokens=[]
    print('Processing started...')
    for i,input_string in enumerate(list(target_docs)):
        string=str.lower(input_string)
        string=remove_punct(string)
        doc = nlp(string)
        lemmatized=[]
        removed_tokens=[]
        for token in doc:
            #(token.text not in spacy_stopwords) was removed, bad idea
            if((not token.is_punct) & (token.text in nlp.vocab)):
                lemmatized.append(token.lemma_.strip())
            elif((token.text not in nlp.vocab)):
                try:
                    if(detect(token.text)=='en'):
                        spell_checked=spell_check(token.text,nlp)
                        if(spell_checked.strip()):
#                             print(detect(token.text),' ', token.text,' ',spell_checked)
                            lemmatized.append(spell_checked)
                    else:
                        removed_tokens.append(token.text)
                except Exception as e:
                    print('failed for: ',token.text,' ',e)
        all_removed_tokens.append(removed_tokens)        
        new_string=" ".join(lemmatized)
        if(len(new_string)<1):
            new_string="unknown"
        new_doc = nlp(new_string)
        cleaned_docs.append(new_string)
        if(not new_doc.has_vector):
            delete_index.append(i)
            print('empty document vector for record ',i,' ', doc)
        else:
            output.append(new_doc.vector)
        if((i+1)%1000==0 and i>0):
            print(i+1,' records processed')
    new_categories=np.delete(target_categories, delete_index,axis=0)
    print('Finished!!')
    return(output,new_categories,cleaned_docs,all_removed_tokens)

def tokenize(text):
    pass


def build_model(estimator):

    scoring={
            'AUC_macro':metrics.make_scorer(metrics.roc_auc_score, average='macro'),
             'AUC_micro':metrics.make_scorer(metrics.roc_auc_score, average='micro'),
             'AUC_weighted':metrics.make_scorer(metrics.roc_auc_score, average='weighted'),
            'Precision_macro':metrics.make_scorer(metrics.precision_score, average='macro'),
            'Precision_micro':metrics.make_scorer(metrics.precision_score, average='micro'),
             'Precision_weighted':metrics.make_scorer(metrics.precision_score, average='weighted'),
            'F1_macro':metrics.make_scorer(metrics.f1_score, average='macro'),
            'F1_micro':metrics.make_scorer(metrics.f1_score, average='micro'),
            'F1_weighted':metrics.make_scorer(metrics.f1_score, average='weighted'),
            'Recall_macro':metrics.make_scorer(metrics.recall_score, average='macro'),
            'Recall_micro':metrics.make_scorer(metrics.recall_score, average='micro'),
            'Recall_weighted':metrics.make_scorer(metrics.recall_score, average='weighted'),
             'ZeroOne':metrics.make_scorer(metrics.zero_one_loss, normalize=True),
             'Cohen':metrics.make_scorer(metrics.cohen_kappa_score),
             'log_loss':metrics.make_scorer(metrics.log_loss),
            }
    params = {
        'min_weight_fraction_leaf': np.linspace(0.02,0.1,20),

    }
        
    model = OneVsRestClassifier(GridSearchCV(estimator, param_grid=params, cv=5, iid=False
                                             ,verbose=1,scoring=scoring,refit='Precision_macro'))
    class new_model():
        def __init__(self,old_model):
            self.model=old_model
            self.is_trained=[]
            self.is_loaded=False
        def train(self,X_train_vec, y_train_vec):
            
            for i in range(0,y_train_vec.shape[1]-1,1):
                print('Training category number: ',i+1)
                self.model.fit(X_train_vec, y_train_vec[:,i]) 
                self.is_trained.append(True)
  
            return(self.model)
        def save(self,filepath):
            if(filepath):
                try:
                    with open(filepath, "wb") as f:
                        cp.dump(self.model, f)
                except Exception as e:
                    print('Failed: ',e)
            else:
                print('invalid path')
        def load(self,filepath):
            if(filepath and os.path.exists(filepath)):
                try:
                    with open(filepath, "rb") as f:
                        self.model=cp.load(f)
                        self.is_loaded=True
                        
                except Exception as e:
                    print('Failed: ',e)        

        def evaluate(self,X_test_vec, y_test_vec, category_names):
            cms=[]
            if(all(self.is_trained) or self.is_loaded):
                for i,category in enumerate(category_names):
                    predicted = self.model.predict(X_test_vec)
                    cm=ConfusionMatrix(y_test_vec[:,i],predicted)
                    cms.append(cm)
                to_include=['Overall ACC','F1 Micro','F1 Macro','PPV Macro','PPV Micro']
                results=pd.DataFrame.from_dict([cm.overall_stat for cm in cms]).loc[:,to_include].replace('None',0).fillna(0)
                results['category_names']=category_names 
                print(results)
                return(results,cms)
            else:
                print('Model has not been trained.')
            
    
    final_model=new_model(model)
    return(final_model)



 

def main():
    random_state=1234
    np.random_state=random_state 
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print('Data loaded from database.')

        cwd=os.getcwd()
        data_folder='/data/'
        data_path=cwd+data_folder
        file_list=['X_train_vec.npy','y_train_vec.npy','X_test_vec.npy','y_test_vec.npy',
           ]
        train_test_vecs=[]
        if (all([os.path.isfile(data_path+file) for file in file_list])):
            print('Found processed data in: '+data_path)
            print('Loading in progress')
            for name in [file.split('.npy')[0] for file in file_list]:
                train_test_vecs.append(np.load(data_path+name+'.npy'))
                print('....   '+data_path+name+'.npy')
            X_train_vec, y_train_vec, X_test_vec, y_test_vec=train_test_vecs
            #only useful for investigating the results
#             all_removed_tokens=np.load(data_path+'all_removed_tokens.npy',allow_pickle=True)
#             X_old_cleaned=np.load(data_path+'X_old_cleaned.npy',allow_pickle=True)
            print('Finished loading processed data')
        else:
            print('Could not find processed data in: '+data_path)
            print('Loading language model ...')
            nlp = spacy.load("en_vectors_web_lg")
            print('Language model loaded.')
            print('Processing data ...')
            X_vec,Y_vec,X_old_cleaned,all_removed_tokens=doc2vec(X,Y,nlp)
            X_train_vec, X_test_vec, y_train_vec, y_test_vec = train_test_split(X_vec, Y_vec, test_size=0.2,random_state=random_state)
            np.save(data_path+'X_vec.npy',X_vec)
            np.save(data_path+'Y_vec.npy',Y_vec)
            np.save(data_path+'X_train_vec.npy',X_train_vec)
            np.save(data_path+'y_train_vec.npy',y_train_vec)
            np.save(data_path+'X_test_vec.npy',X_test_vec)
            np.save(data_path+'y_test_vec.npy',y_test_vec)
            np.save(data_path+'X_old_cleaned.npy',X_old_cleaned)
            np.save(data_path+'all_removed_tokens.npy',all_removed_tokens)
            print('files saved')
        
        print('Building model...')
        estimator=ExtraTreeClassifier(random_state=random_state,class_weight='balanced')
        model = build_model(estimator)
        
        print('Training model...')
        model.train(X_train_vec, y_train_vec)
        
        print('Evaluating model...')
        results,cms=model.evaluate(X_test_vec, y_test_vec, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        model.save(model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()