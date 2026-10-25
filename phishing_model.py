import json
import pandas as pd
from urllib.parse import urlparse
from tld import get_tld, is_tld
import re

# ---------------------------------------------------------------------------
# Brand-impersonation rule (deterministic backup to the Random Forest)
#
# The Random Forest sees only the structure of a URL (length, hyphens, https,
# digits...), so a short, tidy, https link such as hsbc-secure.xyz/verify looks
# ordinary to it. This rule covers that gap: a link is flagged when
#   1. its host contains a known brand name (as a whole word or a prefix, so
#      "chesteracuk" counts but "manchester" does not),
#   2. the host or path contains a lure word (login, verify, secure, nitro...),
#   3. and the link's registered domain is NOT one of that brand's real domains.
# Both lists are short and meant to be edited. Look-alike spellings
# (paypa1, rn for m) and homograph domains are NOT covered.
# ---------------------------------------------------------------------------
BRAND_DOMAINS = {
    'chester':   {'chester.ac.uk', 'sharepoint.com', 'microsoft.com', 'office.com', 'microsoftonline.com'},
    'discord':   {'discord.com', 'discordapp.com', 'discord.gg', 'discord.gift', 'discordstatus.com'},
    'steam':     {'steampowered.com', 'steamcommunity.com'},
    'paypal':    {'paypal.com', 'paypal.me'},
    'hsbc':      {'hsbc.com', 'hsbc.co.uk', 'hsbc.lk'},
    'barclays':  {'barclays.co.uk', 'barclays.com'},
    'lloyds':    {'lloydsbank.com', 'lloydsbank.co.uk', 'lloydsbankinggroup.com'},
    'natwest':   {'natwest.com'},
    'santander': {'santander.co.uk', 'santander.com'},
    'apple':     {'apple.com', 'icloud.com'},
    'microsoft': {'microsoft.com', 'live.com', 'office.com', 'sharepoint.com', 'microsoftonline.com', 'outlook.com'},
    'google':    {'google.com', 'google.co.uk', 'gmail.com', 'youtube.com'},
    'amazon':    {'amazon.com', 'amazon.co.uk', 'amazonaws.com'},
    'netflix':   {'netflix.com'},
    'instagram': {'instagram.com'},
    'facebook':  {'facebook.com', 'fb.com'},
    'roblox':    {'roblox.com'},
}
LURE_WORDS = ('login', 'logon', 'signin', 'secure', 'security', 'verify', 'verification', 'account',
              'update', 'password', 'support', 'auth', 'recovery', 'confirm', 'unlock', 'billing',
              'nitro', 'gift', 'giveaway', 'claim')
# second-level labels used under country codes, e.g. chester.ac.uk, bbc.co.uk
SECOND_LEVEL_LABELS = {'ac', 'co', 'org', 'gov', 'com', 'net', 'edu', 'sch', 'ltd', 'plc'}


class PhishingLinkModel:
	'''
	Phishing Link Detection model

	Retrained as a Random Forest on the PhiUSIIL Phishing URL Dataset
	(Prasad & Chandra, 2023/2024). The trained forest is stored as plain
	JSON (models/phishing_forest.json) rather than a pickled sklearn
	object - this avoids BOTH of the bugs found during development:
	  1. the original MLP needed feature scaling that was never applied
	  2. pickled sklearn/numpy objects break across numpy/sklearn
	     versions (e.g. "No module named 'numpy._core'" when the
	     pickling and loading environments use different numpy majors)
	Since this is just JSON + pure-Python tree traversal, it works
	identically on any Python/numpy/scikit-learn version, forever.
	'''
	def __init__(self):
		with open('models/phishing_forest.json') as file:
			model_data = json.load(file)
		self.trees = model_data['trees']
		self.classes = model_data['classes']

	def _predict_proba_one(self, features, tree):
		node = 0
		while tree['children_left'][node] != -1:
			if features[tree['feature'][node]] <= tree['threshold'][node]:
				node = tree['children_left'][node]
			else:
				node = tree['children_right'][node]
		value = tree['value'][node]
		total = sum(value)
		return [v / total for v in value]

	def _predict_one(self, features):
		best_idx, avg_proba = self._predict_proba(features)
		return self.classes[best_idx]

	def _predict_proba(self, features):
		n_classes = len(self.classes)
		avg_proba = [0.0] * n_classes
		for tree in self.trees:
			proba = self._predict_proba_one(features, tree)
			for i in range(n_classes):
				avg_proba[i] += proba[i]
		avg_proba = [p / len(self.trees) for p in avg_proba]
		best_idx = avg_proba.index(max(avg_proba))
		return best_idx, avg_proba

	def is_phishing(self, text):
		is_phish, _ = self.is_phishing_with_confidence(text)
		return is_phish

	def is_phishing_with_confidence(self, text):
		'''
		Returns (is_phishing, confidence). The confidence (0.0-1.0) is the share
		of the 100 decision trees that agreed; it is None when the verdict came
		from the brand-impersonation rule, which has no model confidence.
		'''
		is_phish, confidence, _ = self.check_text(text)
		return is_phish, confidence

	def check_text(self, text):
		'''
		Full check of every link in a message. Returns
		(is_phishing, model_confidence, impersonated_brand).
		'''
		urls = self.find_urls(text)
		if not urls:
			return False, None, None
		# Normalise: if a URL has no scheme (as commonly typed casually in
		# chat, e.g. "chester.ac.uk" rather than "https://chester.ac.uk"),
		# assume HTTPS by default before feature extraction. Testing found
		# schemeless UK-style compound-TLD domains (.ac.uk, .co.uk) were
		# being misclassified as phishing without this normalisation,
		# since the training data disproportionately represents legitimate
		# URLs with an explicit scheme present.
		urls = [u if u.startswith(('http://', 'https://')) else 'https://' + u for u in urls]

		# 1. deterministic rule: known brand + lure word on an unofficial domain
		for u in urls:
			brand = self.brand_impersonation(u)
			if brand:
				return True, None, brand

		# 2. the Random Forest, scoring EVERY link in the message (a harmless link
		# placed before a malicious one must not hide it). The message counts as
		# phishing if any link is; the confidence is the highest phishing verdict.
		data = self.url_converter(urls)
		results = []
		for _, row in data.iterrows():
			best_idx, avg_proba = self._predict_proba(row.tolist())
			results.append((self.classes[best_idx] == 0, avg_proba[best_idx]))
		phishing_conf = [conf for is_phish, conf in results if is_phish]
		if phishing_conf:
			return True, max(phishing_conf), None
		return False, min(conf for _, conf in results), None

	def find_urls(self, text):
		urls = []
		pattern = r'\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,24})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b'
		# tolerate a stray space after the scheme ("https:// example.com")
		text = re.sub(r'(https?://)\s+', r'\1', text)
		for x in text.split():
			if re.match(pattern, x) and self._has_valid_tld(x):
				urls.append(x)
		return urls

	def _has_valid_tld(self, token):
		'''
		The finder used to accept only 2-6 letter endings, so real endings such as
		.technology or .support (and the reserved .example) were invisible, while
		chat like "ok.thanks" counted as a link. Now: anything with an explicit
		http(s):// is a link; a bare "name.ending" must end in a real TLD.
		'''
		if token.startswith(('http://', 'https://')):
			return True
		host = re.split(r'[/:?#]', token, maxsplit=1)[0]
		if '.' not in host:
			return True  # e.g. an IPv6 literal, matched by another branch of the pattern
		last = host.rsplit('.', 1)[-1]
		if last.isdigit():
			return True  # IPv4 address
		try:
			return bool(is_tld(last))
		except Exception:
			return 2 <= len(last) <= 6  # TLD list unavailable: fall back to the old rule

	def _registered_domain(self, host):
		labels = host.lower().strip('.').split('.')
		if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in SECOND_LEVEL_LABELS:
			return '.'.join(labels[-3:])
		return '.'.join(labels[-2:])

	def brand_impersonation(self, url):
		'''Returns the brand being impersonated, or None. See BRAND_DOMAINS above.'''
		parsed = urlparse(url)
		host = (parsed.hostname or '').lower()
		if not host:
			return None
		host_tokens = [t for t in re.split(r'[^a-z0-9]+', host) if t]
		path_tokens = [t for t in re.split(r'[^a-z0-9]+', (parsed.path or '').lower()) if t]
		if not any(lure in tok for tok in host_tokens + path_tokens for lure in LURE_WORDS):
			return None
		registered = self._registered_domain(host)
		for brand, official in BRAND_DOMAINS.items():
			if any(tok == brand or tok.startswith(brand) for tok in host_tokens) and registered not in official:
				return brand
		return None

	def url_converter(self, urls):
		data = pd.DataFrame()
		data['url'] = pd.Series(urls)

		data['url_length'] = data['url'].apply(lambda x: len(str(x)))
		data['domain'] = data['url'].apply(lambda i: self.process_tld(i))
		feature = ['@', '?', '-', '=', '.', '#', '%', '+', '$', '!', '*', ',', '//']
		for a in feature:
			data[a] = data['url'].apply(lambda i: i.count(a))
		data['abnormal_url'] = data['url'].apply(lambda i: self.abnormal_url(i))
		data['https'] = data['url'].apply(lambda i: self.httpSecure(i))
		data['digits'] = data['url'].apply(lambda i: self.digit_count(i))
		data['letters'] = data['url'].apply(lambda i: self.letter_count(i))
		data['Shortining_Service'] = data['url'].apply(lambda x: self.Shortining_Service(x))
		data['having_ip_address'] = data['url'].apply(lambda i: self.having_ip_address(i))
		X = data.drop(['url', 'domain'], axis=1)

		return X

	def process_tld(self, url):
		try:
			res = get_tld(url, as_object=True, fail_silently=False, fix_protocol=True)
			pri_domain = res.parsed_url.netloc
		except Exception:
			pri_domain = None
		return pri_domain

	def abnormal_url(self, url):
		hostname = urlparse(url).hostname
		hostname = str(hostname)
		match = re.search(hostname, url)
		if match:
			return 1
		else:
			return 0

	def httpSecure(self, url):
		htp = urlparse(url).scheme
		match = str(htp)
		if match == 'https':
			return 1
		else:
			return 0

	def digit_count(self, url):
		digits = 0
		for i in url:
			if i.isnumeric():
				digits = digits + 1
		return digits

	def letter_count(self, url):
		letters = 0
		for i in url:
			if i.isalpha():
				letters = letters + 1
		return letters

	def Shortining_Service(self, url):
		match = re.search(r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
						  r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
						  r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
						  r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
						  r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
						  r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
						  r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
						  r'tr\.im|link\.zip\.net',
						  url)
		if match:
			return 1
		else:
			return 0

	def having_ip_address(self, url):
		match = re.search(
			'(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
			'([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'
			'(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
			'([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'
			'((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'
			'(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
			r'([0-9]+(?:\.[0-9]+){3}:[0-9]+)|'
			r'((?:(?:\d|[01]?\d\d|2[0-4]\d|25[0-5])\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d|\d)(?:\/\d{1,2})?)', url)
		if match:
			return 1
		else:
			return 0


if __name__ == '__main__':
	model = PhishingLinkModel()
	urls = [
		'google.com',
		'https://sliit.lk',
		'https://login.chester.ac.uk/student',
		'http://192.168.0.55/bank/secure/verify?acc=123&pin=4567',
		'http://bit.ly/3xK9zP',
		'http://paypal-secure-login.verify-account.xyz/login',
		'https://login-chester-ac-uk.example/student',
		'https:// hsbc-secure.example/verify',
	]
	for url in urls:
		print(f'{url}:\t{model.check_text(url)}')
