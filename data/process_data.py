import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

class data:
    def __init__(self):
        self.categories=pd.DataFrame()
        self.messages=pd.DataFrame()
        self.messages_filepath=None
        self.categories_filepath=None
        self.clean=pd.DataFrame()
        self.is_loaded=False
        self.is_clean=False

    def check_path(self,path):
        if(os.path.exists(path)):
            return(True)
        else:
            print(path+' is not accessible or does not exist')
            return(False)
    def get_info(target_set):
        """
        Extracts ['column','Missing','Duplicated','Unique','Type'] information about a pandas dataframe
        """
        info=pd.DataFrame([[x,pd.isna(target_set[x]).sum(),target_set[x].duplicated().sum(),
                            len(target_set[x].unique()),target_set[x].dtype] for x in target_set.columns])
        info.columns=['column','Missing','Duplicated','Unique','Type']
        return(info)
        
    def load_data(self,messages_filepath, categories_filepath):
        if(all([self.check_path(path) for path in [messages_filepath, categories_filepath]])):
            try:
                self.categories = pd.read_csv(categories_filepath) 
                self.messages =pd.read_csv(messages_filepath) 
            except Exception as e:
                print(e)
                return(False)
            self.is_loaded=True
            return(True)
        else:
            return(False)

    def clean_data(self):
        try:
            new_categories=self.categories.iloc[:,1:].categories.str.split(';',expand=True).apply(lambda x:x.str.split('-',expand=True)[1],axis=1)
            new_categories=new_categories.apply(lambda x:[int(item) for item in (x.astype(int)==1)])
            new_categories.columns=[item.split('-')[0] for item in self.categories.categories.str.split(';',expand=True).loc[0,:]]
            new_categories['id']=self.categories.iloc[:,0]
            new_df = self.messages.merge(new_categories,on=['id'],how='left')
            self.clean=new_df.loc[-((new_df.message.duplicated()) | (new_df.id.duplicated()))]
            self.is_clean=True
            return(True)
        except Exception as e:
            print(e)
            return(False)
       

    def save_data(self, database_filename):
        try:
            engine = create_engine('sqlite:///'+str(database_filename))
            if(engine.has_table('Main')):
                print('Warning: old data has been replaced')                
            self.clean.to_sql('Main',engine, index=False,if_exists='replace')  
            return(True)
        except Exception as e:
            print(e)
            return(False)        


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        new_data=data()
        if(new_data.load_data(messages_filepath, categories_filepath)):
            print('Files loaded')
            print('Cleaning data...')
            if(new_data.clean_data()):
                print('Data has been cleaned')
                print('Saving data...\n    DATABASE: {}'.format(database_filepath))
                if(new_data.save_data(database_filepath)):
                    print('data has been saved')
                else:
                    print('Failed to save data')
            else:
                print('Failed to clean Data')
        else:
            print('Failed to load files')
        
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()