import pickle
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Backup check for high-severity, statement-chaining SQLi patterns
# (e.g. "'; DROP TABLE users;--"). Testing showed the LSTM model can miss
# these - likely because its tokenizer strips out ';' during preprocessing,
# so it never learns to associate that character with attacks. This regex
# catches that specific gap as a deterministic safety net alongside the ML
# model, rather than relying on the LSTM alone for something this severe.
DANGEROUS_SQL_PATTERN = re.compile(
	r';\s*(drop|delete|truncate|alter|insert|update)\s+', re.IGNORECASE
)


class SQLIModel:
	'''
	SQL Injection detection model
	'''
	def __init__(self):
		self.model = tf.keras.models.load_model("models/sql_lstmv4.h5")
		with open("models/Wtokanizer.pickle", "rb") as f:
			self.data = pickle.load(f)

	def tokanizers(self, input_string):
		input_sequence = self.data.texts_to_sequences([input_string])
		maxlen = 100
		padded_sequences = pad_sequences(input_sequence, maxlen=maxlen)
		pred = self.model.predict(padded_sequences)
		pred = np.int64(pred > 0.5)
		return pred

	def is_sqli_str(self, input_string):
		# deterministic backup check first - catches severe patterns
		# the ML model's tokenizer is known to miss (see comment above)
		if DANGEROUS_SQL_PATTERN.search(input_string):
			return True
		pred = self.tokanizers(input_string)
		return np.any(pred == 1)


if __name__ == '__main__':
	model = SQLIModel()

	# classic SQLi payloads - should return True
	print('or 1  =  1 --', model.is_sqli_str('or 1  =  1 --'))
	print("' OR 1=1--", model.is_sqli_str("' OR 1=1--"))
	print("'; DROP TABLE users;--", model.is_sqli_str("'; DROP TABLE users;--"))
	print("'; DELETE FROM accounts;--", model.is_sqli_str("'; DELETE FROM accounts;--"))

	# ordinary chat messages - should return False.
	# NOTE: the tokenizer's vocab includes common English words that are
	# also SQL keywords (select, or, and, all, as, by, from), so false
	# positives on normal chat are possible - that's what we're checking here.
	print('Hi', model.is_sqli_str('Hi'))
	print('can you select the best option for me', model.is_sqli_str('can you select the best option for me'))
	print('I will wait here or come find you', model.is_sqli_str('I will wait here or come find you'))
	print('please update me on the delivery status', model.is_sqli_str('please update me on the delivery status'))

