from string import punctuation, digits

import numpy as np
import matplotlib.pyplot as plt

### Part I

def hinge_loss(feature_matrix, labels, theta, theta_0):
    l=len(labels)
    hingeloss=np.zeros((l,1))
    for i in range(l):
        eval=labels[i]*(np.dot(theta,feature_matrix[i])+theta_0)
        if eval <=1:
            hingeloss[i]=1-eval
        else:
            hingeloss[i]=0
    totalhingeloss=np.sum(hingeloss)
    avghingeloss=totalhingeloss/l
    
    return avghingeloss
    
        
    """
    Section 1.2
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    raise NotImplementedError

def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    if (label*((np.dot(current_theta,feature_vector))+current_theta_0))<=0:
        current_theta=current_theta+(label*feature_vector)
        current_theta_0=current_theta_0+label
    a=(current_theta,current_theta_0)
            
    return a
    """
    Section 1.3
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    raise NotImplementedError

def perceptron(feature_matrix, labels, T):
    thetazero=0
    feature_matrix=np.asarray(feature_matrix)
    n=feature_matrix.shape[1]
    l=len(labels)
    theta=np.zeros((n))
    for t in range(T):
        for i in range(l):
            feature_vector=feature_matrix[i,:]
            label=labels[i]
            a=perceptron_single_step_update(feature_vector,label,theta,thetazero)
            theta=a[0]
            thetazero=a[1]
    b=(theta,thetazero)
    return b
            
            
    """
    Section 1.4a
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    raise NotImplementedError
    
def average_perceptron(feature_matrix, labels, T):
    thetazero=0
    feature_matrix=np.asarray(feature_matrix)
    n=feature_matrix.shape[1]
    theta=np.zeros((n))
    sumtheta=np.zeros((n))
    sumthetazero=0
    l=len(labels)
    for t in range(T):
        for i in range(l):
            feature_vector=feature_matrix[i,:]
            label=labels[i]
            a=perceptron_single_step_update(feature_vector,label,theta,thetazero)
            theta=a[0]
            thetazero=a[1]
            sumtheta=sumtheta+theta
            sumthetazero=sumthetazero+thetazero
    avgtheta=sumtheta/(n*T)
    avgthetazero=sumthetazero/(n*T)
    b=(avgtheta,avgthetazero)
    return b
    """
    Section 1.4b
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    raise NotImplementedError

def pegasos_single_step_update(feature_vector, label, L, eta, current_theta, current_theta_0):
     
    if (label*(np.dot(current_theta,feature_vector) +current_theta_0))<=1:
        current_theta=((1-(L*eta))*current_theta)+(eta*(label*feature_vector))
        current_theta_0=current_theta_0+(eta*label)
    else:
        current_theta=(1-(L*eta))*current_theta
        current_theta_0=current_theta_0

    a=(current_theta,current_theta_0)
            
    return a
    """
    Section 1.5
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    raise NotImplementedError

def pegasos(feature_matrix, labels, T, L):
    thetazero=0
    feature_matrix=np.asarray(feature_matrix)
    n=feature_matrix.shape[1]
    theta=np.zeros((n))
    k=1
    l=len(labels)
    for t in range(T):
        for i in range(l):
            i=np.random.randint(l)
            eta=1/np.sqrt(k)
            feature_vector=feature_matrix[i]
            label=labels[i]
            a=pegasos_single_step_update(feature_vector, label, L, eta, theta, thetazero)
            k=k+1
            theta=a[0]
            thetazero=a[1]     
    b=(theta,thetazero)
    return b
    """
    Section 1.6
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.
    
    For each update, set learning rate = 1/sqrt(t), 
    where t is a counter for the number of updates performed so far (between 1 
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    raise NotImplementedError

### Part II

def classify(feature_matrix, theta, theta_0):
    numrows=feature_matrix.shape[0]
    pred_labels=np.zeros((numrows,), dtype=np.int)
    for i in range(numrows):
        feature_vector=feature_matrix[i,:]
        if (np.dot(theta,feature_vector)+theta_0) <= 0:
            pred_labels[i]=0-1
        else:
            pred_labels[i]=1 
    return pred_labels
                           
                
    """
    Section 2.8
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is the predicted
    classification of the kth row of the feature matrix using the given theta
    and theta_0.
    """
    raise NotImplementedError

def perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    train_theta, train_theta_0=perceptron(train_feature_matrix, train_labels, T)
   # train_theta=train_thetas[0]
   # train_theta_0=train_thetas[1]
    predictlabels_train=classify(train_feature_matrix,train_theta,train_theta_0)
    predictlabels_val=classify(val_feature_matrix,train_theta,train_theta_0)
    train_accuracy=accuracy(predictlabels_train,train_labels)
    val_accuracy=accuracy(predictlabels_val,val_labels)
    
    ans=(train_accuracy,val_accuracy)
    
    return ans
    
    
    """
    Section 2.9
    Trains a linear classifier using the perceptron algorithm with a given T
    value. The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the perceptron algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    raise NotImplementedError

def average_perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    
    train_thetas=average_perceptron(train_feature_matrix, train_labels, T)
    train_theta=train_thetas[0]
    train_theta_0=train_thetas[1]
    predictlabels_train=classify(train_feature_matrix,train_theta,train_theta_0)
    predictlabels_val=classify(val_feature_matrix,train_theta,train_theta_0)
    train_accuracy=accuracy(predictlabels_train,train_labels)
    val_accuracy=accuracy(predictlabels_val,val_labels)
    
    ans=(train_accuracy,val_accuracy)
    
    return ans
    """
    Section 2.9
    Trains a linear classifier using the average perceptron algorithm with
    a given T value. The classifier is trained on the train data. The
    classifier's accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the average perceptron
            algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    raise NotImplementedError

def pegasos_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T, L):
    train_thetas=pegasos(train_feature_matrix, train_labels, T, L)
    train_theta=train_thetas[0]
    train_theta_0=train_thetas[1]
    predictlabels_train=classify(train_feature_matrix,train_theta,train_theta_0)
    predictlabels_val=classify(val_feature_matrix,train_theta,train_theta_0)
    train_accuracy=accuracy(predictlabels_train,train_labels)
    val_accuracy=accuracy(predictlabels_val,val_labels)
    
    ans=(train_accuracy,val_accuracy)
    
    return ans
    """
    Section 2.9
    Trains a linear classifier using the pegasos algorithm
    with given T and L values. The classifier is trained on the train data.
    The classifier's accuracy on the train and validation data is then
    returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the algorithm.
        L - The value of L to use for training with the Pegasos algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    raise NotImplementedError

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary

def bag_of_words_with_length(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and len(word)>=2:
                dictionary[word] = len(dictionary)
    return dictionary

def bag_of_words_with_bigrams(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index
    #stop_words=np.loadtxt(stopwords)
    for text in texts:
        word_list = extract_words(text)
        lastword=''
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
            bigram=lastword+' '+word
            if bigram not in dictionary:
                dictionary[bigram] = len(dictionary)
            lastword=word
    return dictionary

def bag_of_words_stop_wordsbigramsandlength(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index
    #stop_words=np.loadtxt(stopwords)
    stop_words = []
    for line in open('stopwords.txt','r').readlines():
        stop_words.append(line.strip())
    for text in texts:
        word_list = extract_words(text)
        lastword=''
        newword_list=[]
        for word in word_list:
            if word not in stop_words:
                newword_list.append(word)
        for word in newword_list:
            if word not in dictionary and len(word)>=2:
                dictionary[word] = len(dictionary)
            bigram=lastword+' '+word
            if bigram not in dictionary:
                dictionary[bigram] = len(dictionary)
            lastword=word
    return dictionary

def bag_of_words_bigramsandlength(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index
    #stop_words=np.loadtxt(stopwords)
    for text in texts:
        word_list = extract_words(text)
        lastword=''
        for word in word_list:
            if word not in dictionary and len(word)>=2:
                dictionary[word] = len(dictionary)
            bigram=lastword+' '+word
            if bigram not in dictionary:
                dictionary[bigram] = len(dictionary)
            lastword=word
    return dictionary

def bag_of_words_stop_wordsbigrams(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index
    #stop_words=np.loadtxt(stopwords)
    stop_words = []
    for line in open('stopwords.txt','r').readlines():
        stop_words.append(line.strip())
    for text in texts:
        word_list = extract_words(text)
        lastword=''
        newword_list=[]
        for word in word_list:
            if word not in stop_words:
                newword_list.append(word)
        for word in newword_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
            bigram=lastword+' '+word
            if bigram not in dictionary:
                dictionary[bigram] = len(dictionary)
            lastword=word
    return dictionary
def bag_of_words_stop_wordsUnigram(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index
    stop_words = []
    for line in open('stopwords.txt','r').readlines():
        stop_words.append(line.strip())
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in stop_words:
                dictionary[word] = len(dictionary)
    return dictionary

def bag_of_words_stop_wordsUnigramAndLength(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index
    stop_words = []
    for line in open('stopwords.txt','r').readlines():
        stop_words.append(line.strip())
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in stop_words:
                if len(word)>=2:
                    dictionary[word] = len(dictionary)
    return dictionary

 
 
def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix


def extract_additional_features(reviews):
    """
    Section 3.12
    Inputs a list of string reviews
    Returns a feature matrix of (n,m), where n is the number of reviews
    and m is the total number of additional features of your choice

    YOU MAY CHANGE THE PARAMETERS
    """
    return np.ndarray((len(reviews), 0))

def extract_final_features(reviews, dictionary):
    """
    Section 3.12
    Constructs a final feature matrix using the improved bag-of-words and/or additional features
    """
    bow_feature_matrix = extract_bow_feature_vectors(reviews,dictionary)
    additional_feature_matrix = extract_additional_features(reviews)
    return np.hstack((bow_feature_matrix, additional_feature_matrix))

def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
