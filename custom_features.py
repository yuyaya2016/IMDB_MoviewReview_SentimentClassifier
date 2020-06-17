import os
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import lexicon_reader
import nltk
#nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from lexicalrichness import LexicalRichness
from textstat import textstat


class CustomFeats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def __init__(self, filedir):
        self.feat_names = set()
        lexicon_dir = os.path.join(filedir, 'lexicon')
        self.inqtabs_dict = lexicon_reader.read_inqtabs(os.path.join(lexicon_dir, 'inqtabs.txt'))
        self.swn_dict = lexicon_reader.read_senti_word_net(os.path.join(lexicon_dir, 'SentiWordNet_3.0.0_20130122.txt'))

    def fit(self, x, y=None):
        return self

    @staticmethod
    def word_count(review):
        words = review.split(' ')
        return len(words)

    def pos_count(self, review):
        words = review.split(' ')
        count = 0
        for word in words:
            if word in self.inqtabs_dict.keys() and self.inqtabs_dict[word] == lexicon_reader.POS_LABEL:
                count += 1
        return count

    def neg_count(self, review):
        words = review.split(' ')
        count = 0
        for word in words:
            if word in self.inqtabs_dict.keys() and self.inqtabs_dict[word] == lexicon_reader.NEG_LABEL:
                count += 1
        return count    

    def pos_score_count(self, review):
        words = review.split(' ')
        count = 0
        for word in words:
            if word in self.swn_dict.keys() and self.swn_dict[word][0] > self.swn_dict[word][1]:
                count += 1
        return count
    
    def pos_score(self, review):
        words = review.split(' ')
        pos_score = []
        for word in words:
            if word in self.swn_dict.keys():
                pos_score.append(self.swn_dict[word][0])
        return np.sum(pos_score)
    
    def neg_score(self, review):
        words = review.split(' ')
        neg_score = []
        for word in words:
            if word in self.swn_dict.keys():
                neg_score.append(self.swn_dict[word][1])
        return np.sum(neg_score)    
    
    def pos_adj(self, review):
        allowed_word_types = ["J", "RB"]
        words = review.split(' ')
        tag_words = nltk.pos_tag(words)
        target_words = []
        for w in tag_words:
            if w[1][0] in allowed_word_types:
                target_words.append(w[0])   
        count = 0        
        for word in target_words:
            if word in self.inqtabs_dict.keys() and self.inqtabs_dict[word] == lexicon_reader.POS_LABEL:
                count += 1
        return count
    
    @staticmethod
    def review_score(review):
        regex = r'/(?:[1-9][0-9]*|0)\/[1-9][0-9]*/'
        words = review.split(' ')
        score = []
        for word in words:
            match = re.search(regex, word)
            if match:
                score.append(score)
        return np.sum(score)
    
    @staticmethod
    def top_pos_binary_feature(review):
        target_word = ['excellent']
        threshold = 0
        words = filter(lambda r: r.find(target_word) is not -1, review.split(' '))
        count = len(list(words))
        if count > threshold:
              return 1
        else:
              return 0
    
    @staticmethod
    def vader_pos_score(review):
        words = review.split(' ')
        vader = SentimentIntensityAnalyzer()
        positiv_score = []
        for word in words:
            val = vader.polarity_scores(word)['compound'] 
            if val >= 0.05:
                positiv_score.append(val)
        return np.sum(positiv_score)
    
    @staticmethod             
    def vader_neg_score(review):
        words = review.split(' ')
        vader = SentimentIntensityAnalyzer()
        negativ_score = []
        for word in words:
            val = vader.polarity_scores(word)['compound'] 
            if val <= -0.05:
                negativ_score.append(val)
        return np.sum(negativ_score)
                         
#     @staticmethod        
#     def readability_mean(review):
#         scores = []
#         scores.append(textstat.coleman_liau_index(review))
#         scores.append(textstat.flesch_kincaid_grade(review))
#         return np.mean(scores)
    
#     @staticmethod
#     def lexical_density(review):
#     # measure of textual lexical diversity 
#         if re.search("[a-zA-Z]", review) is None:
#             return 0
#         else:
#             return LexicalRichness(review).mtld(threshold=0.72)    
    
    
    def features(self, review): 
        return {
            'length': len(review),
            'num_sentences': review.count('.'),
            'num_words': self.word_count(review),
            'pos_count': self.pos_count(review),
            'neg_count': self.neg_count(review),
            'pos_score_count': self.pos_score_count(review),
            'pos_score': self.pos_score(review),
#             'neg_score': self.neg_score(review),
            'pos_adj': self.pos_adj(review),
#             'review_score': self.review_score(review),
#             'top_pos_binary_feature': self.top_pos_binary_feature(review),
            'vader_pos_score': self.vader_pos_score(review),
#             'vader_neg_score': self.vader_neg_score(review),
#             'readability_mean': self.readability_mean(review),
#             'lexical_density': self.lexical_density(review)
            }

    def get_feature_names(self):
        return list(self.feat_names)

    def transform(self, reviews):
        feats = []
        for review in reviews:
            f = self.features(review)
            [self.feat_names.add(k) for k in f]
            feats.append(f)
        return feats


def get_custom_vectorizer():
    # Experiment with different vectorizers
    return TfidfVectorizer(sublinear_tf = True, ngram_range=(1,2), min_df = 5, max_df = .1, token_pattern = r"\b\w[\w']+\b") 
#     return CountVectorizer(ngram_range=(1,2),min_df = 5, max_df = .15, token_pattern = r"\b\w[\w']+\b") 
