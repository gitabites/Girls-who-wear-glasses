#!usr/bin/env python

import pandas as pd
import numpy as np 

def main():
#read in training data
	train_data = pd.read_csv('train-neat.csv')
	test_data = pd.read_csv('test-neat.csv')
	test_data.columns = ['Insult', 'Date', 'Comment']
#join the train and test into one big frame
	biginsult = pd.concat([train_data, test_data], ignore_index=True)
	
#vectorize on ngrams
	from sklearn.feature_extraction.text import TfidfVectorizer
	vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
	text_features_ng = vectorizer.fit_transform(biginsult['Comment'])
	text_features_ng.shape

#split biginsult back into test and train; name insult and comment objects 
	new_train = biginsult[:3947]
	new_test = biginsult[3947:]
	insult_train = new_train['Insult'].astype('int')
	comment_train = new_train['Comment']
	insult_test = new_test['Insult'].astype('int')
	comment_test = new_test['Comment']


#train NB classifier to predict insultiness of post in Naive Bayes
	from sklearn.naive_bayes import MultinomialNB
	X = text_features_ng[:3947]
	Y = np.array(insult_train)	
	clf = MultinomialNB()
	model_insult = clf.fit(X, Y)

#cross validate
	from sklearn.cross_validation import cross_val_score
	cross_val_score(model_insult, X.toarray(), Y)	
	model_insult.fit(X.toarray(), Y)

#predict on train
	from sklearn import metrics
	from sklearn.metrics import auc_score
	metrics.confusion_matrix(insult_train, model_insult.predict(X))
	insult_train_predict = model_insult.predict_proba(X)

#vectorize test data
	X_test = vectorizer.transform(comment_test)

#predict on test sdata 
	predicted_insult = model_insult.predict(text_features_test_ng)
	probs = model_insult.predict_proba(text_features_test_ng)

#make new probability column in test_data 
	probsadd = probs[:,0]
	test_data['Prob Insult'] = probsadd
	
#make new submission dataframe	
	submission = test_data.set_index(['Insult'])
	del submission['Date']
	del submission['Comment']
	del submission['ID']
	
#write submission to csv
	submission.to_csv('How Insulting', header=True)

if __name__=="__main__":
    main()

