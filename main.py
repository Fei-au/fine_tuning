from datasets import load_dataset

import re
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('punkt')

dataset = load_dataset("ag_news")

stop_words = set(stopwords.words('english'))


#  1. clean data



#  2. tokenizer

#  3. to pytorch datasets

#  4. fine tuning

#  5. evaluate
