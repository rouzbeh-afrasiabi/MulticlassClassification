# Multiclass Classification - Crisis NLP

### Intoduction:
This repo provides an example of a multi-class classification which can be utilized for catagorizing text messages related to natural disasters. The main goal is to use the model for facilitating humanitarian aid and reducing response time. The dataset used here is a multi-language dataset provided by figure8 (https://www.figure-eight.com/dataset/combined-disaster-response-data/). Other dataset are also available through https://crisisnlp.qcri.org/.



### Instructions:
data/process_data.py:
Performs pre-processing of the data and saves it into a database.

models/train_classifier.py:
Performs data cleaning, tokenization and then feeds the data to the training pipline and saves the trained model.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:5000
