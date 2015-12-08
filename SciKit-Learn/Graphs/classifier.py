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


class ReturnValues(object):
  def __init__(self, y0, y1, y2, y3, y4, y5, y6):
     self.y0 = y0
     self.y1 = y1
     self.y2 = y2
     self.y3 = y3
     self.y4 = y4
     self.y5 = y5
     self.y6 = y6
     
def RunCompare():
    if __name__ == "__main__":
        ErrorProtect = 0
        Protect = 0
        EP2 = 0
        EP3 = 0
        EP4 = 0
        EP5 = 0
        EP6 = 0
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
		
		parameters7 = {'vect__ngram_range': [(1, 1), (1, 2), (1,3)],
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
		grid_search3 = GridSearchCV(pipeline, parameters7, n_jobs=-1)#Linear SVM trigrams
        gs_clf = GridSearchCV(text_clf, parameters2, n_jobs=-1)
        gs_clf2 = GridSearchCV(text_clf, parameters5, n_jobs=-1)
		gs_clf3 = GridSearchCV(text_clf, parameters8, n_jobs=-1)#Multinomial Naive Bayes trigrams
        grid_classification=GridSearchCV(pipeline3, parameters3, n_jobs=-1)
        grid_classification2=GridSearchCV(pipeline3, parameters6, n_jobs=-1)
		grid_classification3=GridSearchCV(pipeline3, parameters9, n_jobs=-1)#SGD Logistic Regression trigrams
        #clf = SGDClassifier(**parameters3).fit(docs_train, y_train)
        grid_search.fit(docs_train, y_train)
        gs_clf.fit(docs_train, y_train)
        grid_classification.fit(docs_train,y_train)
        grid_search2.fit(docs_train, y_train)
        gs_clf2.fit(docs_train, y_train)
        grid_classification2.fit(docs_train,y_train)
		grid_search3.fit(docs_train, y_train)
        gs_clf3.fit(docs_train, y_train)
        grid_classification3.fit(docs_train,y_train)


       
        prediction1=dataset.target_names[grid_search.predict(['Not Excellent movie!'])]
        prediction2=dataset.target_names[gs_clf.predict(['Not Excellent film!'])]
        prediction3=dataset.target_names[grid_classification.predict(['Not Excellent flick!'])]
        # TASK: print the cross-validated scores for the each parameters set
        # explored by the grid search
        print(grid_search.grid_scores_)
        print (gs_clf.grid_scores_)
        print (grid_classification.grid_scores_)
        #print (grid_classification.grid_scores_)
        # TASK: Predict the outcome on the testing set and store it in a variable
        # named y_predicted
        y_predicted = grid_search.predict(docs_test)
        y_predicted2=gs_clf.predict(docs_test)
        y_predicted3=grid_classification.predict(docs_test)
        y_predicted4 = grid_search2.predict(docs_test)
        y_predicted5=gs_clf2.predict(docs_test)
        y_predicted6=grid_classification2.predict(docs_test)
		y_predicted7 = grid_search2.predict(docs_test)
        y_predicted8=gs_clf2.predict(docs_test)
        y_predicted9=grid_classification2.predict(docs_test)
		
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
        return ReturnValues(ErrorProtect,EP1, EP2, EP3, EP4, EP5, EP6)


if __name__ == "__main__":
    HistogramData = []
    Result = []
    Result2 = []
    Result3 = []
    Result4 = []
    Result5 = []
    Result6 = []
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
            NumValues = len(TestValues)

            Count = 0
            Count2 = 0
            Count3 = 0
            Count4 = 0
            Count5 = 0
            Count6 = 0

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
                    
            Result.append(Count/float(NumValues))
            Result2.append(Count2/float(NumValues))
            Result3.append(Count3/float(NumValues))
            Result4.append(Count4/float(NumValues))
            Result5.append(Count5/float(NumValues))
            Result6.append(Count6/float(NumValues))

    Count = 0
    Count2 = 0
    Count3 = 0
    Count4 = 0
    Count5 = 0
    Count6 = 0
    print(Result)
    print(Result2)
    print(Result3)
    print(Result4)
    print(Result5)
    print(Result6)
    for i in range(0,Average):
        Count = Count + Result[i]
        Count2 = Count2 + Result2[i]
        Count3 = Count3 + Result3[i]
        Count4 = Count4 + Result4[i]
        Count5 = Count5 + Result5[i]
        Count6 = Count6 + Result6[i]
    
    Count = Count/10.0
    Count2 = Count2/10.0
    Count3 = Count3/10.0
    Count4 = Count4/10.0
    Count5 = Count5/10.0
    Count6 = Count6/10.0

    N = 3
    ind = np.arange(N)
    width = 0.27

    fig = plt.figure()
    ax = fig.add_subplot(111)
    Unigram = [Count, Count2, Count3]
    Bigram = [Count4, Count5, Count6]


    UniRect = ax.bar(ind, Unigram, width, color='b', align = 'center')
    BiRect = ax.bar(ind+width, Bigram, width, color = 'r', align = 'center')
    
    ax.set_xticks(ind)
    ax.set_xticklabels( ('Linear SVM', 'Naive Bayes', 'SGD LR') )
    ax.legend( (UniRect[0], BiRect[0]), ('Unigram', 'Bigram') )

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

    plt.show()

    
