import sys

import numpy as np
from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def get_tuned_lr(train, dev, features, output_file_path='./lr.png'):
    train_vecs = features.fit_transform(train.reviews)
    dev_vecs = features.transform(dev.reviews)
    train_accuracy_list = list()
    dev_accuracy_list = list()
    cs = np.arange(0.5, 6, .1)  
    cs = [0.5, 1, 5, 10, 100, 1000]
    for c in cs:
        model = LogisticRegression(C=c, 
                                   class_weight = 'balanced', 
                                   penalty = 'l2', 
#                                    solver = 'liblinear',
                                   solver = 'newton-cg', 
                                   multi_class = 'multinomial',
#                                    dual = True,
                                   random_state = 42)
        model.fit(train_vecs, train.labels)
        train_preds = model.predict(train_vecs)
        dev_preds = model.predict(dev_vecs)
        (train_score, dev_score) = (accuracy_score(train.labels, train_preds), accuracy_score(dev.labels, dev_preds))
        print("Train Accuracy:", train_score, ", Dev Accuracy:", dev_score)
        train_accuracy_list.append(train_score)
        dev_accuracy_list.append(dev_score)
    plot(cs, train_accuracy_list, dev_accuracy_list, output_file_path)
    best_model = LogisticRegression(C=cs[np.argmax(dev_accuracy_list)], 
                                    class_weight = 'balanced', 
                                    penalty = 'l2', 
                                    solver = 'newton-cg',
#                                     solver = 'liblinear',
                                    multi_class = 'multinomial',
#                                     dual = True,
                                    random_state = 42)
    print(best_model)
    return get_trained_classifier(train, best_model, features)


def get_tuned_rf(train, dev, features, output_file_path='./rf.png'):
    train_vecs = features.fit_transform(train.reviews)
    dev_vecs = features.transform(dev.reviews)
    train_accuracy_list = list()
    dev_accuracy_list = list()
    n_estimators = np.arange(50, 300, 50)  
    for num_estimator in n_estimators:
        model = RandomForestClassifier(n_estimators=num_estimator,  
#                                        max_depth = 5, 
                                       min_samples_split = 2,
                                       min_samples_leaf = 2,
                                       class_weight = 'balanced', 
                                       max_features = 'sqrt', 
#                                        min_impurity_decrease = 0.01,
                                       oob_score = True,
#                                        warm_start = True,
                                       random_state = 42)
        model.fit(train_vecs, train.labels)
        train_preds = model.predict(train_vecs)
        dev_preds = model.predict(dev_vecs)
        (train_score, dev_score) = (accuracy_score(train.labels, train_preds), accuracy_score(dev.labels, dev_preds))
        print("Train Accuracy:", train_score, ", Dev Accuracy:", dev_score)
        train_accuracy_list.append(train_score)
        dev_accuracy_list.append(dev_score)   
    plot(n_estimators, train_accuracy_list, dev_accuracy_list, output_file_path)
    best_model = RandomForestClassifier(n_estimators=n_estimators[np.argmax(dev_accuracy_list)], 
#                                         max_depth = 5,
                                        min_samples_split = 2,                                        
                                        min_samples_leaf = 2, 
                                        class_weight = 'balanced',
                                        max_features = 'sqrt',
#                                         min_impurity_decrease = 0.01,
                                        oob_score = True,  
#                                         warm_start = True,
                                        random_state = 42)
    print(best_model)
    return get_trained_classifier(train, best_model, features)



if __name__ == "__main__":
    filedir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    print("Reading data")
    train_data, dev_data = get_training_and_dev_data(filedir)
    
    print("Training model")
    lr_with_default = get_trained_classifier(train_data, LogisticRegression(), CountVectorizer())
    rf_with_default = get_trained_classifier(train_data, RandomForestClassifier(),  CountVectorizer())

#     print(lr_with_default.predict(["This movie sucks!", "This movie is great!"]))
#     print(rf_with_default.predict(["This movie sucks!", "This movie is great!"]))


    # Experiment with the parameters in the get_tuned_lr and get_tuned_rf methods
#     print("Tuning model")
    tuned_lr = get_tuned_lr(train_data, dev_data, TfidfVectorizer(min_df = 5,
                                                                  max_df = .15,
                                                                  sublinear_tf = True,
                                                                  ngram_range=(1,2), 
                                                                  token_pattern = r"\b\w[\w']+\b"))
    tuned_rf = get_tuned_rf(train_data, dev_data, TfidfVectorizer(sublinear_tf = True, 
                                                                  ngram_range=(1,2), 
                                                                  token_pattern = r"\b\w[\w']+\b"))



    print("Tuning model")
    tuned_lr = get_tuned_lr(train_data, dev_data, get_custom_features(filedir))
    tuned_rf = get_tuned_rf(train_data, dev_data, get_custom_features(filedir))

    
#     print("Saving model and predictions")
#     save(tuned_lr, filedir, 'lr_default')
#     save(tuned_rf, filedir, 'rf_default')

#     save(tuned_lr, filedir, 'lr_custom')
#     save(tuned_rf, filedir, 'rf_custom')
