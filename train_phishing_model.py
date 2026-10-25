"""
Trains the phishing-link detector and exports it to models/phishing_forest.json,
the file that phishing_model.py loads at runtime.

Usage:
    python train_phishing_model.py <path-to-PhiUSIIL_Phishing_URL_Dataset.csv> [output.json]

Dataset (not included in this repository - download it yourself):
    PhiUSIIL Phishing URL Dataset, UCI Machine Learning Repository
    https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+website
    Prasad, A. and Chandra, S. (2024) Computers & Security, 136, 103545.

Method, in brief:
  1. Read the URL and label columns (label 1 = legitimate, 0 = phishing).
  2. Augment the legitimate URLs. In the raw data every legitimate URL is a bare
     "https://www." homepage, so a model trained on it learns "homepage = safe"
     and rejects ordinary deep links. Augmentation adds realistic paths, drops
     "www." from some URLs and switches some to http, so the model has to learn
     real structural signals instead.
  3. Extract the same 20 features that phishing_model.py extracts at runtime.
  4. Train a 100-tree Random Forest and report accuracy on a held-out 20%.
  5. Export every tree as plain JSON, so the bot does not depend on the
     numpy / scikit-learn version that was used for training.
"""
import json
import random
import re
import sys
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42


# ---- Feature extraction: must match url_converter() in phishing_model.py ----

def abnormal_url(url):
    hostname = str(urlparse(url).hostname)
    return 1 if re.search(hostname, url) else 0


def http_secure(url):
    return 1 if urlparse(url).scheme == 'https' else 0


def digit_count(url):
    return sum(c.isnumeric() for c in url)


def letter_count(url):
    return sum(c.isalpha() for c in url)


def shortening_service(url):
    pattern = (r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
               r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
               r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
               r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|'
               r'db\.tt|qr\.ae|adf\.ly|bitly\.com|cur\.lv|ity\.im|'
               r'q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
               r'prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
               r'link\.zip\.net')
    return 1 if re.search(pattern, url) else 0


def having_ip_address(url):
    pattern = (
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
        r'([0-9]+(?:\.[0-9]+){3}:[0-9]+)|'
        r'((?:(?:\d|[01]?\d\d|2[0-4]\d|25[0-5])\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d|\d)(?:\/\d{1,2})?)')
    return 1 if re.search(pattern, url) else 0


def extract_features(urls):
    data = pd.DataFrame()
    data['url'] = pd.Series(urls)
    data['url_length'] = data['url'].apply(lambda x: len(str(x)))
    for ch in ['@', '?', '-', '=', '.', '#', '%', '+', '$', '!', '*', ',', '//']:
        data[ch] = data['url'].apply(lambda i: i.count(ch))
    data['abnormal_url'] = data['url'].apply(abnormal_url)
    data['https'] = data['url'].apply(http_secure)
    data['digits'] = data['url'].apply(digit_count)
    data['letters'] = data['url'].apply(letter_count)
    data['Shortining_Service'] = data['url'].apply(shortening_service)
    data['having_ip_address'] = data['url'].apply(having_ip_address)
    return data.drop(['url'], axis=1)


# ---- Augmentation of legitimate URLs ----

PATH_TEMPLATES = [
    "/", "/about", "/about-us", "/contact", "/products", "/product/{n}",
    "/blog/{slug}", "/news/{slug}", "/article/{slug}", "/user/{n}",
    "/profile/{slug}", "/watch?v={slug}", "/search?q={slug}",
    "/r/{slug}", "/posts/{n}", "/docs/{slug}", "/wiki/{slug}",
    "/category/{slug}", "/item/{n}", "/download/{slug}", "/support",
    "/login", "/help/{slug}", "/{slug}/{n}", "/store/{slug}",
]
SLUG_WORDS = ["review", "guide", "update", "release", "howto", "story",
              "report", "notes", "overview", "summary", "faq", "details"]


def random_path():
    template = random.choice(PATH_TEMPLATES)
    return template.format(n=random.randint(1, 99999),
                           slug=random.choice(SLUG_WORDS) + "-" + str(random.randint(1, 999)))


def augment_legit_url(url):
    """~65% get a realistic path, ~35% lose 'www.', ~15% switch to http."""
    u = url[:-1] if url.endswith('/') else url
    if random.random() < 0.65:
        u = u + random_path()
    r2 = random.random()
    if r2 < 0.35 and 'www.' in u:
        u = u.replace('www.', '', 1)
    if r2 < 0.15:
        u = u.replace('https://', 'http://', 1)
    return u


# ---- Export to a library-independent format ----

def export_forest(clf, path):
    trees = []
    for est in clf.estimators_:
        t = est.tree_
        trees.append({
            'children_left': t.children_left.tolist(),
            'children_right': t.children_right.tolist(),
            'feature': t.feature.tolist(),
            'threshold': t.threshold.tolist(),
            'value': [row[0].tolist() for row in t.value],
        })
    with open(path, 'w') as f:
        json.dump({'trees': trees, 'classes': clf.classes_.tolist(), 'n_features': int(clf.n_features_in_)}, f)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'models/phishing_forest.json'

    random.seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(csv_path, usecols=['URL', 'label'])
    print(f'Loaded {len(df)} URLs: {df["label"].value_counts().to_dict()} (1 = legitimate, 0 = phishing)')

    urls = [augment_legit_url(u) if lab == 1 else u for u, lab in zip(df['URL'], df['label'])]
    X = extract_features(urls)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1,
                                 random_state=SEED, class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f'Held-out accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(classification_report(y_test, y_pred, target_names=['phishing(0)', 'legitimate(1)']))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

    export_forest(clf, out_path)
    print(f'Exported model to {out_path}')


if __name__ == '__main__':
    main()
