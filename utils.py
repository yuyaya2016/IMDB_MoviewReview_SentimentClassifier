import csv
import os
import pickle
import matplotlib.pyplot as plt
import custom_features
from custom_features import CustomFeats
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_pipeline


class Dataset:
    def __init__(self, data, start_idx, end_idx):
        self.data = data
        self.reviews = [row['Review'] for row in data[start_idx:end_idx]]
        self.labels = [row['Category'] for row in data[start_idx:end_idx]]
        self.vecs = None

def get_training_and_dev_data(filedir, dev_rate=0.2):
    with open(os.path.join(filedir, 'train.csv'), 'r', encoding='utf-8') as csvfile:
        data = [row for row in csv.DictReader(csvfile, delimiter=',')]
        for entry in data:
            with open(os.path.join(filedir, 'train', entry['FileIndex'] + '.txt'), 'r', encoding='utf-8') as reviewfile:
                entry['Review'] = reviewfile.read()
    dev_idx = int(len(data) * (1 - dev_rate))
    return Dataset(data, 0, dev_idx), Dataset(data, dev_idx, len(data))

def get_test_data(filedir, output_file_name):
    testfiledir = os.path.join(filedir, 'test')
    with open(output_file_name, 'w', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=['FileIndex', 'Category'])
        writer.writeheader()
        for filename in sorted(os.listdir(testfiledir), key=lambda x: int(os.path.splitext(x)[0])):
            with open(os.path.join(testfiledir, filename), 'r', encoding='utf-8') as reviewfile:
                fileindex = os.path.splitext(filename)[0]
                review = reviewfile.read()
                yield (fileindex, review)

def write_predictions(filedir, classifier, output_file_name):
    with open(output_file_name, 'w', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=['FileIndex', 'Category'])
        writer.writeheader()
        for (fileindex, review) in get_test_data(filedir, output_file_name):
            prediction = dict()
            prediction['FileIndex'] = fileindex
            prediction['Category'] = classifier.predict([review])[0]
            writer.writerow(prediction)

def get_trained_classifier(data, model, features):
    ppl = make_pipeline(features, model)
    return ppl.fit(data.reviews, data.labels)

def get_custom_features(filedir):
    return FeatureUnion([
        ('custom_feats', make_pipeline(CustomFeats(filedir), DictVectorizer())),
        ('bag_of_words', custom_features.get_custom_vectorizer())
    ])

def save(classifier, filedir, output_file_path):
    with open(output_file_path + ".pkl", 'wb') as f:
        pickle.dump(classifier, f)
    write_predictions(filedir, classifier, output_file_path + "_test.csv")

def load_classifier(input_file_path):
    return pickle.load(open(input_file_path, 'rb'))

def plot(xs, train_accuracy_list, dev_accuracy_list, output_file_path=None):
    plt.clf()
    plt.plot(xs, train_accuracy_list, label='train')
    plt.plot(xs, dev_accuracy_list, label='dev')
    plt.ylabel('Accuracy')
    plt.legend()
    if output_file_path is not None:
        plt.savefig(output_file_path)
    else:
        plt.show()
