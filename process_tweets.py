from csv_helper import CSVHelper
from nltk.tokenize import TweetTokenizer
import re
import string
from stop_words import get_stop_words
from nltk.stem.arlstem import ARLSTem

# for punctuation removal
translator = str.maketrans('', '', string.punctuation)
# for stop word removal
stop_words = get_stop_words('en') # set(stopwords.words('english'))
# for stemming
stemmer = ARLSTem()
# for digit removal
digit = re.compile(r'\d+')

# get tweets from given file
def get_tweets():
    tweets = CSVHelper.load_csv("./data/Tweets_2016London.csv")
    return tweets

# remove punctuation
# input : a sentence (str)
# output : a sentence without punctuation (str)
def remove_punc(sentence):
    return str(sentence).translate(translator).encode('ascii', 'ignore')

# remove url
# input : a sentence (str)
# output : a sentence without url (str)
def remove_url(sentence):
    return re.sub(r"http\S+", "", sentence)

# remove stop word
# input : a word list
# output : a word list without stop word in the list
def remove_stop_word(word_list):
    result = []
    for word in word_list:
        if word.lower() not in stop_words:
            result.append(word.lower())
    return result

# stemming
# input : word list
# output : stemmed word list
def stem(word_list):
    result = []
    for word in word_list:
        result.append(stemmer.stem(word).lower())
    return result

def remove_others(word_list):
    result = []
    for line in word_list:
        if line[0] != '#' and line[0] != '&' and line[0] != '@':
            if digit.match(line) == None:
                result.append(line)
    return result

# tokenize tweets
# input : tweets list
# output : tokenized tweets a nested list
def tokenize_tweets(tweets_list):
    tk = TweetTokenizer()
    result = []
    for line in tweets_list:
        line = remove_url(str(line))
        temp = tk.tokenize(line.lower())
        temp = remove_others(temp)
        temp = remove_stop_word(temp)
        line = words_to_sentence(temp)
        line = remove_punc(line)
        line = line.decode('utf-8')
        temp = tk.tokenize(line.lower())
        temp = remove_stop_word(temp)
        temp = stem(temp)
        result.append(temp)
    return result

# convert word lists to sentence list
# input : a nested word list
# output : a sentence list
def words_to_sentence(word_list):
    result = ''
    for word in word_list:
        result += (str(word) + ' ')
    return result[:-1]

