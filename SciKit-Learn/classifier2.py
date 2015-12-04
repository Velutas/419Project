"""Build a sentiment analysis / polarity model
Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess wether the opinion of the author is
positive or negative.
In this examples we will use a movie review dataset.
"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.linear_model import SGDClassifier


if __name__ == "__main__":
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV

    # the training data folder must be passed as first argument
    movie_reviews_data_folder = sys.argv[1]
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)
    #print (docs_train)
    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent

   
    

    # SGD Logistic MACHINES
    pipeline3 = Pipeline([
    ('vect', TfidfVectorizer(min_df=1, max_df=0.95)),
    ('clf', SGDClassifier(loss='log')),
    ])

    pipeline4 = Pipeline([
    ('vect', TfidfVectorizer(min_df=2, max_df=0.95)),
    ('clf', SGDClassifier(loss='log')),
    ])

    pipeline6 = Pipeline([
    ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
    ('clf', SGDClassifier(loss='log')),
    ])


    
    # SUPPORT VECTOR MACHINES
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=1, max_df=0.95)),
        ('clf', LinearSVC(C=1000)),
    ])
    pipeline2 = Pipeline([
        ('vect', TfidfVectorizer(min_df=2, max_df=0.95)),
        ('clf', LinearSVC(C=1000)),
    ])
    pipeline5 = Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', LinearSVC(C=1000)),
    ])
    

    #print (pipeline)
    #NAIVE BAYES
    text_clf = Pipeline([('vect', TfidfVectorizer(min_df=1, max_df=0.95)),
                      ('clf', MultinomialNB()),
    ])
    text_clf2 = Pipeline([('vect', TfidfVectorizer(min_df=2, max_df=0.95)),
                      ('clf', MultinomialNB()),
    ])
    text_clf3 = Pipeline([('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
                      ('clf', MultinomialNB()),
    ])
    
    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
    }

    parameters2 = {'vect__ngram_range': [(1, 1), (1, 2)],
               'clf__alpha': (1e-2, 1e-3),
    }
    
    parameters3 = {'vect__ngram_range': ((1, 1), (1, 2)),  #bigrams
                   'clf__alpha': (1e-2, 1e-3),
                   #'clf__alpha': (0.00001, 0.000001),
    }

    #Linear SVM with tf=1,2,3
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
    grid_search2 = GridSearchCV(pipeline2, parameters, n_jobs=-1)
    grid_search3 = GridSearchCV(pipeline5, parameters, n_jobs=-1)

    #Multinomial NB with tf=1,2,3
    gs_clf = GridSearchCV(text_clf, parameters2, n_jobs=-1)
    gs_clf2 = GridSearchCV(text_clf2, parameters2, n_jobs=-1)
    gs_clf3 = GridSearchCV(text_clf3, parameters2, n_jobs=-1)

    #SGD Log Reg with tf=1,2,3
    grid_classification=GridSearchCV(pipeline3, parameters3, n_jobs=-1)
    grid_classification2=GridSearchCV(pipeline4, parameters3, n_jobs=-1)
    grid_classification3=GridSearchCV(pipeline6, parameters3, n_jobs=-1)

    #TF-1
    grid_search.fit(docs_train, y_train)
    gs_clf.fit(docs_train, y_train)
    grid_classification.fit(docs_train,y_train)

    #TF-2
    grid_search2.fit(docs_train, y_train)
    gs_clf2.fit(docs_train, y_train)
    grid_classification2.fit(docs_train,y_train)

    #TF-3
    grid_search3.fit(docs_train, y_train)
    gs_clf3.fit(docs_train, y_train)
    grid_classification3.fit(docs_train,y_train)

    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search
    print(grid_search.grid_scores_)
    print (gs_clf.grid_scores_)
    print (grid_classification.grid_scores_)
    #print (grid_classification.grid_scores_)
    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted

    #TF-1
    y_predicted = grid_search.predict(docs_test)
    y_predicted2=gs_clf.predict(docs_test)
    y_predicted3=grid_classification.predict(docs_test)

    #TF-2
    y_predicted4 = grid_search2.predict(docs_test)
    y_predicted5=gs_clf2.predict(docs_test)
    y_predicted6=grid_classification2.predict(docs_test)

    #TF-3
    y_predicted7 = grid_search3.predict(docs_test)
    y_predicted8=gs_clf3.predict(docs_test)
    y_predicted9=grid_classification3.predict(docs_test)
    
    # Print the classification report
    #Linear SVM 1,2,3
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))
    print(metrics.classification_report(y_test, y_predicted4,
                                        target_names=dataset.target_names))
    print(metrics.classification_report(y_test, y_predicted7,
                                        target_names=dataset.target_names))

    #Multinomail NB 1,2,3
    print(metrics.classification_report(y_test, y_predicted2,
                                        target_names=dataset.target_names))
    print(metrics.classification_report(y_test, y_predicted5,
                                        target_names=dataset.target_names))
    print(metrics.classification_report(y_test, y_predicted8,
                                        target_names=dataset.target_names))

    #SGD Log Reg 1,2,3
    print(metrics.classification_report(y_test, y_predicted3,
                                        target_names=dataset.target_names))
    print(metrics.classification_report(y_test, y_predicted6,
                                        target_names=dataset.target_names))
    print(metrics.classification_report(y_test, y_predicted9,
                                        target_names=dataset.target_names))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print("Linear SVM-1:", cm)
    cm4 = metrics.confusion_matrix(y_test, y_predicted4)
    print("Linear SVM-2:", cm4)
    cm7 = metrics.confusion_matrix(y_test, y_predicted7)
    print("Linear SVM-3:", cm7)
    
    cm2 = metrics.confusion_matrix(y_test, y_predicted2)
    print("MultinomialNB-1:", cm2)
    cm5 = metrics.confusion_matrix(y_test, y_predicted5)
    print("MultinomialNB-2:", cm5)
    cm8 = metrics.confusion_matrix(y_test, y_predicted8)
    print("MultinomialNB-3:", cm8)

    cm3 = metrics.confusion_matrix(y_test, y_predicted3)
    print("SGD LG-1:", cm3)
    cm6 = metrics.confusion_matrix(y_test, y_predicted6)
    print("SGD LG-2:", cm6)
    cm9 = metrics.confusion_matrix(y_test, y_predicted9)
    print("SGD LG-3:", cm9)


    import matplotlib.pyplot as plt
    plt.matshow(cm)
    plt.show()

    import matplotlib.pyplot as plt2
    plt2.matshow(cm2)
    plt2.show()

    import matplotlib.pyplot as plt3
    plt3.matshow(cm3)
    plt3.show()
