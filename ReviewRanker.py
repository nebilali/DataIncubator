'''
	Author: Nebil Ali
	Desc:	Amazon Review Model
	Date: 	5/1/2016
'''
######################################################
# Imports
######################################################
import pandas as pd
import numpy as np
import gzip
import datetime
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tfidf import *
######################################################
# Code
######################################################
gnb = GaussianNB()
logreg = LogisticRegression()

def parse(path):
	g = gzip.open(path, 'rb')
	for l in g:
		yield eval(l)

def getDF(path):
	i = 0
	df = {}
	dic ={}
	dic[1] = 0
	dic[2] = 0
	dic[3] = 0
	dic[4] = 0
	dic[5] = 0
	
	for d in parse(path):
		if int(d['reviewTime'].split(',')[1]) > 2008:
			if dic[d['overall']] <= 1000: 
				dic[d['overall']] += 1
				df[i] = d
				i += 1
				if i >= 5000:
					break
	print i
	return pd.DataFrame.from_dict(df, orient='index')

stopwords = ['a', 'an', 'and', 'are', 'as', 'be', 'by', 'for', 'from', 'has', 'he', 'i', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']
df = pd.read_pickle('short_reviews.pkl')

def make_rank_PDF_graph(df):
	# get most common word given Ranking
	star_count = {}
	for star in range(1,6):
		word_count = Counter(" ".join(df[df.overall == star]["summary"].str.lower()).split())
		for word in stopwords:
			if word in word_count:
				word_count.pop(word, None)
		count = word_count.most_common(200)
		count.sort(key=lambda x: x[1])
		count.reverse()
		star_count[star] = count

	# find most common words
	counts = [(k,v)for i in range(1,6) for k,v in star_count[i]]
	counts.sort(key=lambda x: x[1])
	dic_t = {}
	count = 0
	for k, v in counts:
		if count >= 200:
			break
		if k in dic_t:
			pass
		else:
			dic_t[k] = v
			count += 1
	keys = dic_t.keys()
	corpus = keys


	star = 1
	p = {}
	colors = ['r', 'b', 'g', '#624ea7', 'c']
	for star in range(1,6):
		num_words = 200
		# df = df[df.overall == star]
		words_per = df[df.overall==star]["reviewText"].str.lower()
		# corpus = Counter(" ".join(words_per).split()).most_common(num_words)
		# corpus = [k for k,v in corpus]
		words_per = words_per
		word_num = {word: i for i, word in enumerate(corpus)}
		axis = word_num.values()
		numerator = np.zeros([num_words, num_words])
		denominator = np.zeros([num_words, num_words])
		denominator += 0.1

		count = 0
		df_str = df["reviewText"].str.lower()
		for word2 in corpus:
			print count
			count += 1
			for line in words_per: 
				if word2 in line : 
					numerator[word_num[word2]] += 1.0
			for line in df_str:
				if word2 in line : 
					denominator[word_num[word2]] += 1.0			

		Z = np.divide(numerator, denominator)
		p[star] = plt.plot(range(num_words), Z, color = colors[star-1])

	plt.legend([p[i][0] for i in range(1,6)], ('1 star','2 stars', '3 stars', '4 start', '5 stars'))
	plt.ylabel('Probability')
	plt.xlabel('Words Numbered')
	plt.title('50 Most Frequency Words in Review Summary by Rating')
	plt.tight_layout()
	plt.title("Probability Distribution given word per Rank")
	plt.show()
# make_rank_PDF_graph(df)

def word_count_plot(df):
	# get most common word given Ranking
	star_count = {}
	for star in range(1,6):
		word_count = Counter(" ".join(df[df.overall == star]["summary"].str.lower()).split())
		for word in stopwords:
			if word in word_count:
				word_count.pop(word, None)
		count = word_count.most_common(50)
		count.sort(key=lambda x: x[1])
		count.reverse()
		star_count[star] = count

	# find most common words
	counts = [(k,v)for i in range(1,6) for k,v in star_count[i]]
	counts.sort(key=lambda x: x[1])
	dic_t = {}
	count = 0
	for k, v in counts:
		if count >= 50:
			break
		if k in dic_t:
			pass
		else:
			dic_t[k] = v
			count += 1
	keys = dic_t.keys()

	# filter out duplicates between rankings
	star_dict = {}
	for star in range(1, 6):
		word_count = Counter(" ".join(df[df.overall == star]["summary"].str.lower()).split())
		d = dict.fromkeys(keys, 0)
		for key in word_count:
		 if key in keys:
			d[key] = word_count[key]
		star_dict[star] = d

	# Make stacked Bar Plot
	colors = ['r', 'b', 'g', '#624ea7', 'c']
	p = {}
	for i in range(1, 6):
		p[i-1] = plt.bar(range(len(star_dict[i])), star_dict[i].values(), color=colors[i-1])
	plt.legend([p[i][0] for i in range(5)], ('1 star', '2 stars', '3 stars', '4 start', '5 stars'))
	plt.xticks(range(len(star_dict[i])), star_dict[i].keys(), rotation='vertical')
	plt.ylabel('Word Count')
	plt.title('50 Most Frequency Words in Review Summary by Rating')
	plt.tight_layout()
	plt.show()
# word_count_plot(df)

def vectorize_sample(feature_space, sample):
	vector = np.zeros([len(feature_space)])
	for word, index in feature_space.items():
		if word in sample:
			vector[index] = 1
	return vector

def createX(df):
	corpus = [[word for text in df[df.overall == rank]["reviewText"] for word in text.lower().split() if word not in stopwords] for rank in range(1, 6)]
	words = []
	for rank in range(1, 6):
		sample = [text.lower().split() for text in df[df.overall == rank]["reviewText"]]
		top_tfidf_words = get_tfidf_topk(sample, corpus, 1000)
		words.extend([word for word, score in top_tfidf_words])

	feature_space = create_feature_space(words)
	num_features = len(feature_space)
	X = np.zeros([len(df), num_features])

	for row in range(len(df)):
		X[row, :] = vectorize_sample(feature_space, df["reviewText"][row])
	
	return X

######################################################
# Script
######################################################
X = createX(df)
Y = df.overall
model_gnb = gnb.fit(X, Y)
model_logreg = logreg.fit(X, Y)
Y_pred_gnb = model_gnb.predict(X)
Y_pred_logreg = model_logreg.predict(X)

log_mean = []
gnb_mean = []
log_std = []
gnb_std = []
p = {}
for rank in range(1, 6):
	mask = Y == rank
	length = sum(mask)
	pred_gnb = np.zeros([length])
	pred_logreg = np.zeros([length])
	count = 0
	for i, bol in enumerate(mask):
		if bol:
			pred_gnb[count] = Y_pred_gnb[i]
			pred_logreg[count] = Y_pred_logreg[i]
			count += 1
	gnb_mean.append(pred_gnb.mean())
	gnb_std.append(pred_gnb.std())
	log_mean.append(pred_logreg.mean())
	log_std.append(pred_logreg.std())

	print "gnb overall:    ", rank, accuracy_score(Y_pred_gnb, Y)
	print "logreg overall: ", rank, accuracy_score(Y_pred_logreg, Y)
	print "gnb:    ", rank, accuracy_score(pred_gnb, Y[mask])
	print "logreg: ", rank, accuracy_score(pred_logreg, Y[mask])
	print ""

print gnb_mean
print gnb_std
print log_mean
print log_std

peakval_gnb = [str(val) for val in gnb_mean]
peakval_log = [str(val) for val in log_mean]
colors = ['r', 'b', 'g', '#624ea7', 'c']
width = 0.35
plt.grid()
position_gnb = [i-width/2 for i in range(1, 6)]
position_log = [i+width/2 for i in range(1, 6)]
p[0] = plt.bar(position_gnb, gnb_mean, width, align="center", color=colors[0], yerr=gnb_std)
p[1] = plt.bar(position_log, log_mean, width, align="center", color=colors[1], yerr=log_std)
plt.legend([p[i][0] for i in range(2)], ('gnb mean', 'log mean'), loc=4)
plt.ylabel('Predicted Rank')
plt.xlabel('Actual Rank')
plt.title('GNB vs. Logistic Regression Average Prediction and Std Deviation')
plt.tight_layout()
plt.show()