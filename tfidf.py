##############################################################
# Imports
##############################################################
import nltk
from os import listdir
from collections import defaultdict
from collections import Counter
import math
import operator

##############################################################
# Support Functions
##############################################################
def get_all_files(directory):
	return [f for f in listdir(directory)]

def standardize(raw_excerpt):
	return [tok.encode('utf-8') for tok in nltk.word_tokenize(raw_excerpt.lower().decode('utf-8'))]

def load_file_excerpts(file_path):
	f = open(file_path)
	excerpts = f.readlines()
	f.close()
	return [standardize(exc) for exc in excerpts]

def load_directory_excerpts(dir_path):
	files = get_all_files(dir_path)
	return [excerpts for f in files for excerpts in load_file_excerpts(dir_path + '/' + f)]

def flatten(list_of_list): 
	return [val for sublist in list_of_list for val in sublist]

##############################################################
# TFIDF
##############################################################
'''
	sample and corpus is list of list, each list is a document split into list of words
	sample contains all documents of one topic/group
	corpus contains all documents
'''
def get_tf(sample):
	sample = flatten(sample)
	cnt = Counter(sample)
	return cnt

def get_df(words, corpus):
	dic = defaultdict(int)
	for excerpt in corpus:
		word_counts = Counter(excerpt)
		for word in word_counts:
			# dic[word] += word_counts[word]
			dic[word] += 1
	return dic

def get_idf(corpus):
	words = set(flatten(corpus))
	dic = defaultdict(float)
	N = float(len(corpus))
	df_dic = get_df(words, corpus)
	for word in words:
		df = df_dic[word]
		dic[word] = math.log(N / df)
	return dic

def get_tfidf(tf_dict, idf_dict):
	dic = defaultdict(float)
	for word in idf_dict:
		if word in tf_dict:
			dic[word] = float(idf_dict[word]) * tf_dict[word]
		else:
			dic[word] = 0.0
	return dic

def get_tfidf_weights_topk(tf_dict, idf_dict, k):
	tfidf = get_tfidf(tf_dict, idf_dict)
	return list(sorted(tfidf.iteritems(), key=operator.itemgetter(1), reverse=True)[:k])

def get_tfidf_topk(sample, corpus, k):
	tf = get_tf(sample)
	idf = get_idf(corpus)
	return get_tfidf_weights_topk(tf, idf, k)

##############################################################
# Support Functions
##############################################################
def create_feature_space(wordlist):
	wordlist = set(wordlist)
	features = defaultdict(int)
	n = 0
	for word in wordlist:
		features[word] = n
		n += 1
	return features