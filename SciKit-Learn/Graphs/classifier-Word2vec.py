"""Build a sentiment analysis / polarity model
Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess wether the opinion of the author is
positive or negative.
In this examples we will use a movie review dataset.
"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import matplotlib.pyplot as plt
import numpy as np

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
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale

class ReturnValues(object):
  def __init__(self, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9):
     self.y0 = y0
     self.y1 = y1
     self.y2 = y2
     self.y3 = y3
     self.y4 = y4
     self.y5 = y5
     self.y6 = y6
     self.y7 = y7
     self.y8 = y8
     self.y9 = y9
     
def RunCompare():
    if __name__ == "__main__":
        ErrorProtect = 0
        Protect = 0
        EP2 = 0
        EP3 = 0
        EP4 = 0
        EP5 = 0
        EP6 = 0
        EP7 = 0
        EP8 = 0
        EP9 = 0
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

        n_dim = 300
        #Initialize model and build vocab
        imdb_w2v = Word2Vec(size=n_dim, min_count=10)
        imdb_w2v.build_vocab(docs_train)

        #Train the model over train_reviews (this may take several minutes)
        imdb_w2v.train(docs_train)
        def buildWordVector(text, size):
          vec = np.zeros(size).reshape((1, size))
          count = 0.
          for word in text:
              try:
                  vec += imdb_w2v[word].reshape((1, size))
                  count += 1.
              except KeyError:
                  continue
          if count != 0:
              vec /= count
          return vec
        train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in docs_train])
        train_vecs = scale(train_vecs)

        #Train word2vec on test tweets
        imdb_w2v.train(docs_test)
        
        #Build test tweet vectors then scale
        test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in docs_test])
        test_vecs = scale(test_vecs)

        # SGD Logistic MACHINES
        pipeline3 = Pipeline([
        
        ('clf', SGDClassifier(loss='log')),
        ])

        
        # SUPPORT VECTOR MACHINES
        pipeline = Pipeline([
            
            ('clf', LinearSVC(C=1000)),
        ])

        #print (pipeline)
        #NAIVE BAYES
        text_clf = Pipeline([
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

        # TRIGRAMS
        parameters7 = {'vect__ngram_range': [(1, 1), (1, 2), (1,3)]
                       }

        parameters8 = {'vect__ngram_range': [(1, 1), (1, 2), (1,3)],
                       'clf__alpha': (1e-2, 1e-3),
        }

        parameters9 = {'vect__ngram_range': ((1, 1), (1, 2), (1,3)),  # unigrams or bigrams
                       'clf__alpha': (1e-2, 1e-3),
                       #'clf__alpha': (0.00001, 0.000001),
        }

         
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
        grid_search2 = GridSearchCV(pipeline, parameters4, n_jobs=-1)
        grid_search3 = GridSearchCV(pipeline, parameters7, n_jobs=-1) #Linear SVM trigrams
        gs_clf = GridSearchCV(text_clf, parameters2, n_jobs=-1)
        gs_clf2 = GridSearchCV(text_clf, parameters5, n_jobs=-1)
        gs_clf3 = GridSearchCV(text_clf, parameters8, n_jobs=-1)#Multinomial Naive Bayes trigrams
        grid_classification=GridSearchCV(pipeline3, parameters3, n_jobs=-1)
        grid_classification2=GridSearchCV(pipeline3, parameters6, n_jobs=-1)
        grid_classification3=GridSearchCV(pipeline3, parameters9, n_jobs=-1)#SGD Logistic Regression trigrams

        #clf = SGDClassifier(**parameters3).fit(docs_train, y_train)
        grid_search.fit(train_vecs, y_train)
        gs_clf.fit(train_vecs, y_train)
        grid_classification.fit(train_vecs,y_train)

        grid_search2.fit(train_vecs, y_train)
        gs_clf2.fit(train_vecs, y_train)
        grid_classification2.fit(train_vecs,y_train)
        grid_search3.fit(train_vecs, y_train)
        gs_clf3.fit(train_vecs, y_train)
        grid_classification3.fit(train_vecs,y_train)


       
       
        #print (grid_classification.grid_scores_)
        # TASK: Predict the outcome on the testing set and store it in a variable
        # named y_predicted
        y_predicted = grid_search.predict(test_vecs)
        y_predicted2=gs_clf.predict(test_vecs)
        y_predicted3=grid_classification.predict(test_vecs)

        y_predicted4 = grid_search2.predict(test_vecs)
        y_predicted5=gs_clf2.predict(test_vecs)
        y_predicted6=grid_classification2.predict(test_vecs)
        y_predicted7 = grid_search2.predict(test_vecs)
        y_predicted8=gs_clf2.predict(test_vecs)
        y_predicted9=grid_classification2.predict(test_vecs)
		
        # Print the classification report
        print(metrics.classification_report(y_test, y_predicted,
                                            target_names=dataset.target_names))
        ErrorProtect = y_test
        EP1 = y_predicted

        print(metrics.classification_report(y_test, y_predicted4,
                                            target_names=dataset.target_names))
        EP4 = y_predicted4

        print(metrics.classification_report(y_test, y_predicted2,
                                            target_names=dataset.target_names))
        EP2 = y_predicted2

        print(metrics.classification_report(y_test, y_predicted5,
                                            target_names=dataset.target_names))
        EP5 = y_predicted5

        print(metrics.classification_report(y_test, y_predicted3,
                                            target_names=dataset.target_names))
        EP3 = y_predicted3

        print(metrics.classification_report(y_test, y_predicted6,
                                            target_names=dataset.target_names))
        EP6 = y_predicted6

        print(metrics.classification_report(y_test, y_predicted7,
                                            target_names=dataset.target_names))
        EP7 = y_predicted7

        print(metrics.classification_report(y_test, y_predicted8,
                                            target_names=dataset.target_names))
        EP8 = y_predicted8

        print(metrics.classification_report(y_test, y_predicted9,
                                            target_names=dataset.target_names))
        EP9 = y_predicted9

        # Print and plot the confusion matrix
        cm = metrics.confusion_matrix(y_test, y_predicted)
        print(cm)
        print(prediction1)
        print (prediction2)
        print(prediction3)
        cm2 = metrics.confusion_matrix(y_test, y_predicted2)
        print(cm2)

        cm3 = metrics.confusion_matrix(y_test, y_predicted3)
        print(cm3)
##        if ErrorProtect != 0:
        return ReturnValues(ErrorProtect,EP1, EP2, EP3, EP4, EP5, EP6, EP7, EP8, EP9)


if __name__ == "__main__":
    HistogramData = []
    Result = []
    Result2 = []
    Result3 = []
    Result4 = []
    Result5 = []
    Result6 = []
    Result7 = []
    Result8 = []
    Result9 = []
    Average = 10

    for i in range(0,Average):
        HistogramData.append(RunCompare())

    for i in range(0,Average):
        if HistogramData[i] != None:
            print("\n")
            print("\n")
            TestValues = HistogramData[i].y0
            Method1 = HistogramData[i].y1
            Method2 = HistogramData[i].y2
            Method3 = HistogramData[i].y3
            Method4 = HistogramData[i].y4
            Method5 = HistogramData[i].y5
            Method6 = HistogramData[i].y6
            Method7 = HistogramData[i].y7
            Method8 = HistogramData[i].y8
            Method9 = HistogramData[i].y9
            NumValues = len(TestValues)

            Count = 0
            Count2 = 0
            Count3 = 0
            Count4 = 0
            Count5 = 0
            Count6 = 0
            Count7 = 0
            Count8 = 0
            Count9 = 0

            for i in range(0,NumValues):
                if TestValues[i] == Method1[i] :
                    Count = Count + 1
                if TestValues[i] == Method2[i]:
                    Count2 = Count2 + 1
                if TestValues[i] == Method3[i]:
                    Count3 = Count3 + 1
                if TestValues[i] == Method4[i]:
                    Count4 = Count4 + 1
                if TestValues[i] == Method5[i]:
                    Count5 = Count5 + 1
                if TestValues[i] == Method6[i]:
                    Count6 = Count6 + 1
                if TestValues[i] == Method7[i]:
                    Count7 = Count7 + 1
                if TestValues[i] == Method8[i]:
                    Count8 = Count8 + 1
                if TestValues[i] == Method9[i]:
                    Count9 = Count9 + 1
                    
            Result.append(Count/float(NumValues))
            Result2.append(Count2/float(NumValues))
            Result3.append(Count3/float(NumValues))
            Result4.append(Count4/float(NumValues))
            Result5.append(Count5/float(NumValues))
            Result6.append(Count6/float(NumValues))
            Result7.append(Count7/float(NumValues))
            Result8.append(Count8/float(NumValues))
            Result9.append(Count9/float(NumValues))

    Count = 0
    Count2 = 0
    Count3 = 0
    Count4 = 0
    Count5 = 0
    Count6 = 0
    Count7 = 0
    Count8 = 0
    Count9 = 0
    print(Result)
    print(Result2)
    print(Result3)
    print(Result4)
    print(Result5)
    print(Result6)
    print(Result7)
    print(Result8)
    print(Result9)
    for i in range(0,Average):
        Count = Count + Result[i]
        Count2 = Count2 + Result2[i]
        Count3 = Count3 + Result3[i]
        Count4 = Count4 + Result4[i]
        Count5 = Count5 + Result5[i]
        Count6 = Count6 + Result6[i]
        Count7 = Count7 + Result7[i]
        Count8 = Count8 + Result8[i]
        Count9 = Count9 + Result9[i]
    
    Count = Count/10.0
    Count2 = Count2/10.0
    Count3 = Count3/10.0
    Count4 = Count4/10.0
    Count5 = Count5/10.0
    Count6 = Count6/10.0
    Count7 = Count7/10.0
    Count8 = Count8/10.0
    Count9 = Count9/10.0

    N = 3
    ind = np.arange(N)
    width = 0.27

    fig = plt.figure()
    ax = fig.add_subplot(111)
    Unigram = [Count, Count2, Count3]
    Bigram = [Count4, Count5, Count6]
    Trigram = [Count7, Count8, Count9]


    UniRect = ax.bar(ind, Unigram, width, color='b', align = 'center')
    BiRect = ax.bar(ind+width, Bigram, width, color = 'r', align = 'center')
    TriRect = ax.bar(ind+width*2, Trigram, width, color = 'k', align = 'center')
    
    ax.set_xticks(ind)
    ax.set_xticklabels( ('Linear SVM', 'Naive Bayes', 'SGD LR') )
    ax.legend( (UniRect[0], BiRect[0], TriRect[0]), ('Unigram', 'Bigram', 'Trigram') )

    plt.grid(True)
    plt.ylabel("Accuracy Percentage")
    plt.xlabel("Sentiment Analysis Methods")
    plt.title("Comparison of Accuracy between Sentiment Analysis Methods \n on Movie Review Data")

    def autolabel(rects):
      for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

    autolabel(UniRect)
    autolabel(BiRect)
    autolabel(TriRect)

    plt.show()

    
