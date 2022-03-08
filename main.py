#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#we start importing neccessary packages from apis.
#I will use tensorflow API for the AI.
#thanks


# In[16]:


import nltk
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import random
import tflearn
import tensorflow as tf
import json #we will store out json file for resposes
import pickle

with open("intents.json") as file:
    response_data = json.load(file)


# In[10]:


#print(data) #just checking.


# In[17]:


try:
    #load in some pickle data.
    with open("data.pickle","rb") as file:
        words,labels,training,output = pickle.load(file)
except:
    #we don't want to run data over and over waste or time and resources.
    words = [] #list of words
    labels = []
    docs = []
    tgs = [] #short for tags

    for intent in response_data["intents"]:
        for pattern in intent["patterns"]:
            #tokenize words, taking each word and putting them as a pattern. I am converting string object to tokenized words
            #so that python will understand.
            wrds = nltk.word_tokenize(pattern)
            #we want to save these tokenized pattern in our word list
            words.extend(wrds)
            #add to the documents list the pattern of words.
            docs.append(wrds)
            #we can tell is pattern = tag for better accuracy of AI.
            tgs.append(intent["tag"])

            #now we can use conditional branching for proper AI response.
            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    #gonna stem all the words in this words lists and remove any duplicate elements.
    #We want to see how many words it has seen already.

    #w.lower() is required as you want all your text to be lowercase for effeciency and accuracy.
    words = [stemmer.stem(w.lower()) for w in words]

    #this does 3 things
    #set(words) = removes all duplicate words
    #list() = converts set back to list
    #sorted() = sorts words for cleaner code.

    words = sorted(list(set(words)))
    #sort labels.
    labels = sorted(labels)

    #a bag of words 
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]


    for x, doc in enumerate(docs):
        bag = [] #get that bag

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1) #we say that word exists and should go here.
            else:
                bag.append(0) #no words, give 0.

        #we add tags list of labels for 1 so we can add it to hot bags
        output_row = out_empty[:]
        output_row[labels.index(tgs[x])] = 1

        training.append(bag)
        output.append(output_row)
    
    #we store out training data and output data to numpy as a list.
    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle","wb") as file:
        pickle.dump((words,labels,training,output), file)


# In[18]:


tf.compat.v1.reset_default_graph()


# In[22]:


#define the input shape we are expecting for the model
net = tflearn.input_data(shape=[None,len(training[0])])

#in this case words are tokenized and run  through a neural network with 16 nodes and 
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


# In[25]:


#checking if model is already created 
try:
    model.load("model.tflearn")
#model is then saved for work in the future.
except:
    model.fit(training,output,n_epoch = 1000,batch_size = 8,show_metric=True)
    model.save("model.tflearn")


# In[29]:


#creating a function that takes key words from words lists taken from sentence.
def bag_ofwords(sen,words):
    #check bag if this element of words exists or it doesnt.
    bag = [0 for _ in range(len(words))]
    
    #we have to tokenize each word and add to our bag.
    s_words = nltk.tokenize(sen)
    s_words = [stemmer.stem(word.lower()) for word in s_words] #this is stemming our words in tokenized sentences.
    
    for sn in s_words:
        for i,w in enumerate(words):
            if w == sn:
                bag[i] = 1 #append a 1 if word exists in bag of words.
            #else append(0), we already appended 0 on all list.
            
    return numpy.array(bag)


# In[31]:


#chat function user end.
def chat():
    print("Hello, I am a real human *wink*. How may I help you today? (type exit to quit) ")
    while True:
        inp = input("You: ")
        if inp.lower() == "exit":
            break
        
        
        res = model.predict([bag_ofwords(inp,words)])
        print(res)

chat()



