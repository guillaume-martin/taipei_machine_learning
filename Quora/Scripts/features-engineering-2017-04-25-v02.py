''' generate a .csv file whith all features for the Quora case
    author: Guillaume
    date: 2014-04-25
    version: v02

    changes:
    adding tdidf vectorization
'''

#import sys, getopt
import pandas as pd
import nltk, string
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

INPUTFILE = '../Data/train.csv'
OUTPUTFILE = '../Data/features-2017-04-25-v03.csv'

df = pd.read_csv(INPUTFILE)


# count characters in questions 1 and 2
# then calculare the difference
df['char_count1'] = df['question1'].apply(lambda x: len(x))
df['char_count2'] = df['question2'].apply(lambda x: len(str(x)))
df['char_diff'] = abs(df['char_count1'] - df['char_count2'])

# make bins
df['char_diff_bins'] = pd.cut(df['char_diff'], 100)

# make dummy varaibles on char_diff_bins
# and drop the column with bins
dummies = pd.get_dummies(df['char_diff_bins'], prefix='char_diff')
df.drop('char_diff_bins', axis=1, inplace=True)
df.join(dummies)

# count the number of words in each question
# then calculare the difference
df['tokens1'] = df['question1'].apply(lambda x: nltk.word_tokenize(x))
df['word_count1'] = df['tokens1'].apply(lambda x: len(x))
df['tokens2'] = df['question2'].apply(lambda x: nltk.word_tokenize(str(x)))
df['word_count2'] = df['tokens2'].apply(lambda x: len(x))
df['word_diff'] = abs(df['word_count1'] - df['word_count2'])

# make bins
df['word_diff_bins'] = pd.cut(df['word_diff'], 20)

# make dummy varaibles from bins
# the drop the bins column
dummies = pd.get_dummies(df['word_diff_bins'], prefix='word_diff')
df.drop('word_diff_bins', axis=1, inplace=True)
df.join(dummies)


# compare the question words
# we extract the question word from each question
# and look if the 2 questions have the same question word
question_words = ['what','where','when','why','who','which','whose','how']
def get_qword(question):
    ''' extract the question word from a string
    parameters:
    ----------
        question    string
            the sentence the question words needs
            to be extracted from
    return:
    -------
        word    string
            a question word
    '''

    question = str(question).lower()

    for word in question_words:
        if word in question:
            return word

def compare_qwords(row):
    ''' compares qword1 and qword 2 in a given row
        and returns 1 if they are different and
        0 otherwise
    parameters:
    -----------
        row
            a row of a dataframe
    return:
    -------
        integer
            1 if the question words are different,
            o otherwise
    '''

    if row['qword1'] != row['qword2']:
        return 1
    else:
        return 0

df['qword1'] = df['question1'].apply(lambda x: get_qword(x))
df['qword2'] = df['question2'].apply(lambda x: get_qword(x))
df['qword_diff'] = df.apply(lambda row: compare_qwords(row), axis=1)


# shared words
# we count how many unique words are common to the 2 questions
def share_words_count(q1, q2):
    ''' counts how many words are common to 2 questions
    parameters:
    ----------
        q1,q2    string
            the 2 questions of the pair
    return:
    -------
        integer
            the number of words that appear in both questions
    '''

    # convert both strings to all lowercase
    q1 = q1.lower()
    q2 = q2.lower()

    # tokenize the sentences without the punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    q1_tokens = tokenizer.tokenize(q1)
    q2_tokens = tokenizer.tokenize(q2)

    # remove duplicate words in each listimport nltk, string
    set1 = set(q1_tokens)
    set2 = set(q2_tokens)

    # return the number of elements that appear in both sets
    # & is intersection of 2 sets
    return len(set1 & set2)

df['shared_words_count'] = df.apply(lambda row: share_words_count(str(row['question1']), str(row['question2'])), axis=1)


# shared percentage
# calculate the percentage of shared words over
# the total of unique words in the question pair
def shared_pct(row):
    unique_words = list(set(row['tokens1'] + row['tokens2']))
    unique_words_count = len(unique_words)
    pct_shared = (row['shared_words_count'] / unique_words_count) * 100

    return pct_shared

df['pct_shared'] = df.apply(lambda row: shared_pct(row), axis=1)
df['pct_shared_bins'] = pd.cut(df['pct_shared'],10)


# make dummies for percentage shared
dummies = pd.get_dummies(df['pct_shared_bins'], prefix='pct_shared')
df.drop('pct_shared', axis=1, inplace=True)
df = df.join(dummies)


# evaluate similarity of pair
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
        return ((tfidf * tfidf.T).A)[0,1]
    except:
        return 0

df['cosine'] = df.apply(lambda row: cosine_sim(row['question1'], row['question2']), axis=1)


# remove useless columns
useless_col = ['id', 'qid1', 'qid2', 'question1', 'question2', 'tokens1', 'tokens2']
df.drop(useless_col, axis=1, inplace=True)

# save the dataframe
df.to_csv(OUTPUTFILE)

