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
    ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
    ('clf', SGDClassifier(loss='log')),
    ])

    
    # SUPPORT VECTOR MACHINES
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', LinearSVC(C=1000)),
    ])

    #print (pipeline)
    #NAIVE BAYES
    text_clf = Pipeline([('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
                      ('clf', MultinomialNB()),
    ])

    parameters = {'vect__ngram_range': [(1, 1)],
    }

    parameters2 = {'vect__ngram_range': [(1, 1)],
               'clf__alpha': (1e-2, 1e-3),
    }
    
    parameters3 = {'vect__ngram_range': [(1, 1)],  # unigrams or bigrams
                   'clf__alpha': (1e-2, 1e-3),
                   #'clf__alpha': (0.00001, 0.000001),
    }
    
    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    parameters4 = {'vect__ngram_range': [(1, 1), (1, 2)],
    }

    parameters5 = {'vect__ngram_range': [(1, 1), (1, 2)],
               'clf__alpha': (1e-2, 1e-3),
    }
    
    parameters6 = {'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                   'clf__alpha': (1e-2, 1e-3),
                   #'clf__alpha': (0.00001, 0.000001),
    }

     
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)#Linear SVM unigrams
    grid_search2 = GridSearchCV(pipeline, parameters4, n_jobs=-1)#Linear SVM bigrams

    
    gs_clf = GridSearchCV(text_clf, parameters2, n_jobs=-1)#Multinomial Naive Bayes unigrams
    gs_clf2 = GridSearchCV(text_clf, parameters5, n_jobs=-1)#Multinomial Naive Bayes bigrams
    
    grid_classification=GridSearchCV(pipeline3, parameters3, n_jobs=-1)#SGD Logistic Regression unigrams
    grid_classification2=GridSearchCV(pipeline3, parameters6, n_jobs=-1)#SGD Logistic Regression bigrams
    
    #unigrams
    grid_search.fit(docs_train, y_train)#LinearSVM
    gs_clf.fit(docs_train, y_train)#Multinomial NB
    grid_classification.fit(docs_train,y_train)#SGD Log Reg

    #bigrams
    grid_search2.fit(docs_train, y_train)#LinearSVM
    gs_clf2.fit(docs_train, y_train)#Multinomial NB
    grid_classification2.fit(docs_train,y_train)#SGD Log Reg


    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search
    print(grid_search.grid_scores_)
    print (gs_clf.grid_scores_)
    print (grid_classification.grid_scores_)
    #print (grid_classification.grid_scores_)
    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted

    #unigrams
    y_predicted = grid_search.predict(docs_test)
    y_predicted2=gs_clf.predict(docs_test)
    y_predicted3=grid_classification.predict(docs_test)

    #bigrams
    y_predicted4 = grid_search2.predict(docs_test)
    y_predicted5=gs_clf2.predict(docs_test)
    y_predicted6=grid_classification2.predict(docs_test)

    
    # Print the classification report
    #unigram LinearSVM
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    #bigram LinearSVM
    print(metrics.classification_report(y_test, y_predicted4,
                                        target_names=dataset.target_names))

    #unigram Multinomial NB
    print(metrics.classification_report(y_test, y_predicted2,
                                        target_names=dataset.target_names))

    #bigram Multinomial NB
    print(metrics.classification_report(y_test, y_predicted5,
                                        target_names=dataset.target_names))

    #unigram SGD Log Reg
    print(metrics.classification_report(y_test, y_predicted3,
                                        target_names=dataset.target_names))

    #bigram SGD Log Reg
    print(metrics.classification_report(y_test, y_predicted6,
                                        target_names=dataset.target_names))

    
    # Print and plot the confusion matrix

    #unigram LinearSVM
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    #bigram LinearSVM
    cm4 = metrics.confusion_matrix(y_test, y_predicted4)
    print(cm4)

    #unigram MultinomialNB
    cm2 = metrics.confusion_matrix(y_test, y_predicted2)
    print(cm2)

    #bigram MultinomialNB
    cm5 = metrics.confusion_matrix(y_test, y_predicted5)
    print(cm5)

    #unigram SGD Log Reg    
    cm3 = metrics.confusion_matrix(y_test, y_predicted3)
    print(cm3)

    #bigram SGD Log Reg
    cm6 = metrics.confusion_matrix(y_test, y_predicted3)
    print(cm6)

    user=""
    while user != 'Q':
        user=input("Enter a phrase to predict: ")
        prediction1=dataset.target_names[grid_search.predict([user])]
        prediction2=dataset.target_names[gs_clf.predict([user])]
        prediction3=dataset.target_names[grid_classification.predict([user])]
        prediction4=dataset.target_names[grid_search2.predict([user])]
        prediction5=dataset.target_names[gs_clf2.predict([user])]
        prediction6=dataset.target_names[grid_classification2.predict([user])]
        print("Linear SVM Unigram: ",prediction1)
        print("Linear SVM Bigram: ",prediction4)
        print ("Multinomial NB Unigram: ",prediction2)
        print("Multinomial NB Bigram: ",prediction5)
        print("SGD Log Reg Unigram: ",prediction3)
        print("SGD Log Reg Unigram: ",prediction6)
        
        


    
    import matplotlib.pyplot as plt
    plt.matshow(cm)
    plt.show()

    import matplotlib.pyplot as plt2
    plt2.matshow(cm2)
    plt2.show()

    import matplotlib.pyplot as plt3
    plt3.matshow(cm3)
    plt3.show()

   
