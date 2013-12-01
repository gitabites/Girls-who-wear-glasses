Girls-who-wear-glasses
======================

Kaggle insult contest

A solution to Impermium's Kaggle contest: "Detecting Insults in Social Commentary"

The insult data is split into a training set and a test set. The training data consists of forum comments, along with their dates and insult/non-insult score.
If the comment is an insult, it gets a 1; if not, a 0. The test data has dates and comments, but no score. 

The prediction code is in insultpythonscript. It uses pandas and the TfidfVectorizer from the scikit learn library. 
