import json
import joblib
import re
import nltk
import pickle
import string
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


class MeanEmbeddingVectorizer():
	def __init__(self, word2vec):
		self.word2vec = word2vec
		self.dim = len(next(iter(word2vec.values())))

	def transform(self, X):
		return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0) for words in X])

	def fit(self, X, y):
		return self


class CBModel:
	'''
	Cyber Bullying / hate speech identification model
	'''
	'''
	0: age
	1: ethnicity
	2: gender
	3: not_cyberbullying
	4: other_cyberbullying
	5: religion
	'''
	def __init__(self):
		self.model = joblib.load('models/cb_hatSpeech.joblib')
		self.porter = PorterStemmer()
		self.wl = WordNetLemmatizer()
		self.tokenizer = TweetTokenizer(preserve_case=False,
										strip_handles=True,
										reduce_len=True)
		with open('models/mevectorizer.pickle', 'rb') as file:
			self.mev = pickle.load(file)
		self._load_hostile_terms('models/hostile_lexicon.json')

	def initial_cleaning(self, text):
		text = re.sub(r"^RT[\s]+", "", text)
		text = text.lower()
		text = re.sub(r"https?:\/\/.*[\r\n]*", "https", text)
		text = re.sub(r"@[A-Za-z0-9]+", "", text)
		text = re.sub(r'[0-9]', ' ', text)
		text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
		text = re.sub(r'\s+', ' ', text)
		return text

	def remove_stopword(self, text):
		a = [i for i in text.split() if i not in stopwords.words('english')]
		return ' '.join(a)

	def stem(self, text):
		a = [self.porter.stem(i) for i in text.split()]
		return ' '.join(a)

	def get_wordnet_pos(self, treebank_tag):
		"""
		return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
		"""
		if treebank_tag.startswith('J'):
			return wordnet.ADJ
		elif treebank_tag.startswith('V'):
			return wordnet.VERB
		elif treebank_tag.startswith('N'):
			return wordnet.NOUN
		elif treebank_tag.startswith('R'):
			return wordnet.ADV
		else:
			return wordnet.NOUN

	def lemmatizer(self, text):
		word_pos_tags = nltk.pos_tag(self.tokenizer.tokenize(text))
		a = [self.wl.lemmatize(tag[0], self.get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)]
		return " ".join(a)

	def text_preprocessing(self, text):
		return self.lemmatizer(self.stem(self.remove_stopword(self.initial_cleaning(text))))

	def predict(self, text):
		clean_text = self.text_preprocessing(text)
		tokens = clean_text.split()
		# IMPORTANT: MeanEmbeddingVectorizer.transform() expects a LIST of
		# tokenized documents (list of word-lists), not a raw string.
		# Passing a bare string here would make it iterate character-by-character,
		# producing a wrong-shaped, near-meaningless embedding.
		input_vec = self.mev.transform([tokens])
		pred = self.model.predict(input_vec)
		return pred

	# ------------------------------------------------------------------
	# Two-stage decision (classifier + hostile-vocabulary evidence)
	#
	# Testing on labelled data the classifier was NOT trained on showed it
	# has very little discriminating power outside its training set: it
	# flagged 85-90% of harmless comments from Wikipedia, YouTube and Kaggle
	# sources, and 30 of 40 harmless chat-style messages. It also treats
	# texting spellings ("ur", "im") as strong signals of harassment - a
	# style shortcut learned from the tweet training data. Confidence
	# thresholds cannot fix this because the model is confidently wrong
	# ("ur" alone -> 0.97 confidence).
	#
	# The decision therefore requires independent evidence: a message is
	# flagged only if (1) it contains at least one term from a small,
	# reviewed vocabulary of hostile words derived from labelled data across
	# several platforms (models/hostile_lexicon.json), AND (2) the classifier
	# does not judge the message to be not_cyberbullying. The classifier still
	# supplies the harassment category used in moderation logs.
	# ------------------------------------------------------------------
	def _load_hostile_terms(self, path):
		with open(path, 'r', encoding='utf-8') as f:
			self.hostile_terms = set(json.load(f))

	def has_hostile_term(self, text):
		words = self.initial_cleaning(text).split()
		return any(w in self.hostile_terms for w in words)

	def predict_with_confidence(self, text):
		'''
		Returns (class_label, confidence, number_of_recognised_words) for the
		classifier's top class.
		'''
		clean_text = self.text_preprocessing(text)
		tokens = clean_text.split()
		known = [t for t in tokens if t in self.mev.word2vec]
		proba = self.model.predict_proba(self.mev.transform([tokens]))[0]
		idx = int(np.argmax(proba))
		return int(self.model.classes_[idx]), float(proba[idx]), len(known)

	def is_cyber_bullying(self, text):
		try:
			if len(text) < 3:
				return False
			if not self.has_hostile_term(text):
				return False  # no independent evidence of hostility
			label, confidence, n_known = self.predict_with_confidence(text)
			return label != 3  # classifier does not veto (3 = not_cyberbullying)
		except ValueError:
			return False


if __name__ == '__main__':
	cb_model = CBModel()
	text = "fuck you"
	print(text, cb_model.is_cyber_bullying(text))
	text = "you bitch"
	print(text, cb_model.is_cyber_bullying(text))
	text = 'https://www.google.com'
	print(text, cb_model.is_cyber_bullying(text))
	text = 'hey how are you doing today'
	print(text, cb_model.is_cyber_bullying(text))
	for text in ['can i have ur number', 'love u', 'im so sad', 'hi what s going on', 'how are you', 'hey']:
		print(text, cb_model.is_cyber_bullying(text), '(should be False)')
