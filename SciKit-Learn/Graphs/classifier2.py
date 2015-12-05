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
     
def autolabel(rects):
      for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

def RunCompare():
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
        ErrorProtect = y_test
        Lin1 = y_predicted
        Lin2 = y_predicted4
        Lin3 = y_predicted7
        
        #Multinomail NB 1,2,3
        print(metrics.classification_report(y_test, y_predicted2,
                                            target_names=dataset.target_names))
        print(metrics.classification_report(y_test, y_predicted5,
                                            target_names=dataset.target_names))
        print(metrics.classification_report(y_test, y_predicted8,
                                            target_names=dataset.target_names))

        NB1 = y_predicted2
        NB2 = y_predicted5
        NB3 = y_predicted8
        
        #SGD Log Reg 1,2,3
        print(metrics.classification_report(y_test, y_predicted3,
                                            target_names=dataset.target_names))
        print(metrics.classification_report(y_test, y_predicted6,
                                            target_names=dataset.target_names))
        print(metrics.classification_report(y_test, y_predicted9,
                                            target_names=dataset.target_names))
        SGD1 = y_predicted3
        SGD2 = y_predicted6
        SGD3 = y_predicted9


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

        return ReturnValues(ErrorProtect,Lin1, Lin2, Lin3, NB1, NB2, NB3, SGD1, SGD2, SGD3)
    



# Create the Graphs
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
            Lin1 = HistogramData[i].y1
            Lin2 = HistogramData[i].y2
            Lin3 = HistogramData[i].y3
            NB1 = HistogramData[i].y4
            NB2 = HistogramData[i].y5
            NB3 = HistogramData[i].y6
            SGD1 = HistogramData[i].y7
            SGD2 = HistogramData[i].y8
            SGD3 = HistogramData[i].y9
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
                if TestValues[i] == Lin1[i] :
                    Count = Count + 1
                if TestValues[i] == Lin2[i]:
                    Count2 = Count2 + 1
                if TestValues[i] == Lin3[i]:
                    Count3 = Count3 + 1
                if TestValues[i] == NB1[i]:
                    Count4 = Count4 + 1
                if TestValues[i] == NB2[i]:
                    Count5 = Count5 + 1
                if TestValues[i] == NB3[i]:
                    Count6 = Count6 + 1
                if TestValues[i] == SGD1[i]:
                    Count7 = Count7 + 1
                if TestValues[i] == SGD2[i]:
                    Count8 = Count8 + 1
                if TestValues[i] == SGD3[i]:
                    Count9 = Count9 + 1
                    
            Result.append(Count/float(NumValues))
            Result2.append(Count2/float(NumValues))
            Result3.append(Count3/float(NumValues))
            Result4.append(Count4/float(NumValues))
            Result5.append(Count5/float(NumValues))
            Result6.append(Count6/float(NumValues))
            Result7.append(Count6/float(NumValues))
            Result8.append(Count6/float(NumValues))
            Result9.append(Count6/float(NumValues))

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
    TimesFound1 = [Count, Count4, Count7]
    TimesFound2 = [Count2, Count5, Count8]
    TimesFound3 = [Count3, Count6, Count9]


    TF1 = ax.bar(ind, TimesFound1, width, color='b', align = 'center')
    TF2 = ax.bar(ind+width, TimesFound2, width, color = 'r', align = 'center')
    TF3 = ax.bar(ind+width*2, TimesFound3, width, color = 'k', align = 'center')
    
    ax.set_xticks(ind)
    ax.set_xticklabels( ('Linear SVM', 'Naive Bayes', 'SGD LR') )
    ax.legend( (TF1[0], TF2[0], TF3[0]), ('TF = 1', 'TF = 2', 'TF = 3') )

    plt.grid(True)
    plt.ylabel("Accuracy Percentage")
    plt.xlabel("Sentiment Analysis Methods")
    plt.title("Comparison of Accuracy between Sentiment Analysis Methods \n on Movie Review Data for different values of TF")

    autolabel(TF1)
    autolabel(TF2)
    autolabel(TF3)

    plt.show()
