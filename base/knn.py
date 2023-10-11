import numpy as np
import collections,base.swaptest
from sklearn import metrics

def encode(xss):
    amplitudes=np.sqrt(np.einsum('ij,ij->i',xss,xss))
    amplitudes[amplitudes==0] = 1
    normalised_data=xss/amplitudes[:,np.newaxis]
    return normalised_data
def get_major_vote(labels):
    count=collections.Counter(labels)
    return count.most_common(1)[0][0]
def sapxepmangkhoangcach(mangkc):
    new=sorted(range(len(mangkc)),key=lambda k:mangkc[k])
    new.reverse()
    return new
def laykthaklonnhhat(xs,ys,k):
    mang=[]
    i=0
    for j in ys:
         if(i<=k):
            mang.append(xs[j])
            i=i+1
         else:
            break
    return mang

def distances(vector,dataset,iteration: int = 1):
    distances=[]
    for i in dataset:
       distances.append(base.swaptest.do_muc_do(vector1=i,vector2=vector,iteration=iteration))
    return distances
def predict(traindata,trainlabels,testdatas,k:int=1,iteration:int = 1):
    predict_labels=[]
    for test_data in testdatas:
        kc=distances(test_data,traindata,iteration=iteration)
        dasapxep=sapxepmangkhoangcach(kc)
        x=laykthaklonnhhat(trainlabels,dasapxep,k=k)
        predict_labels.append(get_major_vote(x))
    return predict_labels

def bench_mark(ground_truth, predict):
    """Return predict labels QKNN algorithm

    Args:
        - ground_truth (numpy array 1D): truth labels
        - predict (numpy array 1D): predict labels

    Returns:
        - Tuple: benchmark on classifer problem
    """
    accuracy = metrics.accuracy_score(ground_truth, predict)
    precision = metrics.precision_score(ground_truth, predict, average="weighted")
    recall = metrics.recall_score(ground_truth, predict, average="weighted")
    f1 = metrics.f1_score(ground_truth, predict, average="micro")
    matrix = metrics.confusion_matrix(ground_truth, predict)
    return accuracy, precision, recall, matrix