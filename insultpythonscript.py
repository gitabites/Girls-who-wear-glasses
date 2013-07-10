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

#split biginsult back into test and train 
	new_train = biginsult[:3947]
	new_test = biginsult[3947:]

#train NB classifier to predict insultness of post in Naive Bayes
	from sklearn.naive_bayes import MultinomialNB
	X = text_features_ng[:3947]
	Y = np.array(new_train['Insult'])	
	clf = MultinomialNB()
	model_insult = clf.fit(X, Y)

#cross validate
	from sklearn.cross_validation import cross_val_score
	cross_val_score(model_insult, X.toarray(), Y)	
	model_insult.fit(X.toarray(), Y)

#vectorize test data
	text_features_test_ng = vectorizer.transform(new_test['Comment'])

#predict on test sdata (my question is should a value go into predict proba, eg 1 for insult? or should the whole Y go in?)
	predicted_insult = model_insult.predict(text_features_test_ng)
	probs = model_insult.predict_proba(text_features_test_ng)
#get AUC #can't get this to work
#from sklearn.metrics import auc_score
#insults = np.array(new_test['Insult'])
#scores = []
#scores.append(auc_score(insultf, probs[:, 1]), pos_label=1)
#print("score: %f" % scores[-1])

#zip together ids and insult probabilities for comment, insult
	iden = new_test['Insult']
	probs = model_insult.predict_proba(text_features_test_ng)
	columns = ['ID', 'Insult Prob']
	submission = pd.DataFrame(iden, probs, columns=columns)
	
	
	for id, insult in zip(comment, predicted_insult):
		print '%r => %s' % (id, )


if __name__=="__main__":
    main()

