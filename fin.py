import os
from flask import Flask, g, render_template, flash, request , url_for, session, jsonify ,redirect, session , escape
import pandas as pd
import nltk
import re
import json
import pickle
import numpy as np
import itertools

from flask_mysqldb import MySQL
import csv


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

app= Flask(__name__)

app.config['MYSQL_HOST']= 'localhost'
app.config['MYSQL_USER']= 'root'
app.config['MYSQL_PASSWORD']=''
app.config['MYSQL_DB']='moodcoach'

mysql=MySQL(app)

@app.route('/')
def indext():

	cur=mysql.connection.cursor()
	cur.execute('SELECT text from text_input where id= (select MAX(id) from text_input) ')
	rv= cur.fetchall()
	

	with open("newa.csv","w") as csv_file:
	#csv_file=csv.writer(open("output.csv","w"))
		csv_writer=csv.writer(csv_file)
		csv_writer.writerow(['text'])
		csv_writer.writerows(rv)

	return str(rv)


@app.route('/predict',methods=['GET','POST'])

def apicall():
    #return ("hi")
	def cleaning(text):

		txt=str(text)
		txt=re.sub(r"http\S+","",txt)
		if len(txt) ==0:
			return('no text')
		else:
			txt= txt.split()
			index=0
			for j in range(len(txt)):
				if(txt[j][0])=='@':
					index=j

			txt=np.delete(txt,index)
			if len(txt)==0:
				return('no text')
			else:
				words=txt[0]
				for k in range(len(txt)-1):
					words+= " " + txt[k+1]
				txt=words
				txt= re.sub(r'[^\w]',' ',txt)
				if len(txt)==0:
					return 'no text'
				else:
					txt =''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))
					txt=txt.replace("'","")
					txt =nltk.tokenize.word_tokenize(txt)
					for j in range(len(txt)):
						txt[j]= lem.lemmatize(txt[j],"v")
					if len(txt)==0:
						return 'no text'
					else:
						return txt


	tst_data= pd.read_csv('newa.csv')
	tst_data['text']= tst_data['text'].map(lambda x: cleaning(x))



	tst_data=tst_data.reset_index(drop=True)

	for i in range(len(tst_data)):
		words= tst_data.text[i][0]
		for j in range(len(tst_data.text[i])-1):
			words+= ' ' + tst_data.text[i][j+1]
		tst_data.text[i]= words

	#vectorizer = TfidfVectorizer(min_df=3, max_df=0.9)
	#vectorizer = vectorizer
	#l_test=tst_data.text
	#test_vectors = vectorizer.transform(l_test)

	with open('vectorizer.pickle','rb') as vectorizer:
		loaded_pk =pickle.load(vectorizer)
	#vectorizer = pickle.load(open("vectorizer.pickle"), "rb")
	l_test=tst_data.text
	test_vectors = loaded_pk.transform(l_test)
	#selector = pickle.load(open("selector.pickle"), "rb")

	clf = 'filename'

	if tst_data.empty:
		return(bad_request())
	else:
		with open(clf,'rb') as f:
			loaded_model= pickle.load(f)

			#vectorizer = TfidfVectorizer(min_df=3, max_df=0.9)	
			#l_test=tst_data.text
			#test_vectors = vectorizer.transform(l_test)
			#return(l_test)

		predictions= loaded_model.predict(test_vectors)
		#print(predicted_sentiment)

		prediction_series = list(pd.Series(predictions))

		final_predictions = pd.DataFrame(list(prediction_series))

		responses = jsonify(predictions=final_predictions.to_json(orient="records"))
		#jsonString=jsonString.replace("\"0\":", "\"nameofe\":");
		responses.status_code = 200

		return (responses)

if __name__=="__main__":
	app.run(debug = True)


