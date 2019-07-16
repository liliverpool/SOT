# -*- coding: utf-8 -*-
"""
Created on Mon March 12 18:11:20 2019

@author: slab
"""

import numpy as np
import math
import time
import os

model_name = "f_SOT"

# parameter settings and initializations
start = 9 # haperparameter of TOPIC-NUM K
L = 15   # parameter of L
data_clip = 500 # a parameter to select a part of given data to train
stop_file = open('stopwords2.txt', 'r')
readtext = stop_file.read()
stop_list = readtext.split('\n')
data = []
word_index = dict()
index_word = dict()

# the number of documnets, topics, words and other hyper paramters for dirichlet distribution
docs_num = 1
topic_num = start
words_num = 1
alpha = 0.5
beta = 0.5
eta = 1
iteration_num = 50 # the number of iterations

# all used distributions
topic_word = 0*np.ones([1, 1])
topic_word_list = 0*np.ones([1, 1, 1])
doc_topic = 0*np.ones([1, 1])
docs_list = []
doc_topic_distributions = []
topic_word_distributions = []
topic_word_distribution = []
perplexities = []
eta_list =  []
res_list1 = []
res_list2 = []
per_list = []
# time record 
st= 0
ed= 0
total_time = 0

# create dictionary for training data
def create_dictionary(data):
    global word_index, index_word
    for doc in data:
        for w in doc:
            if w not in word_index:
                word_index[w] = len(word_index)
    index_word = dict(zip(word_index.values(), word_index.keys()))


# KL divergence
def asymmetricKL(P, Q):  
    summ = 0
    for i in range(0, len(P)):
        summ += P[i] * math.log(1 / (Q[i] + 0.0000001))
    return summ  
 
def symmetricalKL(P,Q):  
    return (asymmetricKL(P,Q)+asymmetricKL(Q,P))/2.00

def Diff(P, Q):
    summ = 0
    for i in range(0, len(P)):
         summ += abs(P[i] - Q[i]) + delta
    return abs(base**(summ / len(P)))

# compute the parameter of etas
def compute_eta_list(L):
    global doc_topic_distributions, epsilon, eta_list
    for i in range(L, len(doc_topic_distributions)):
        d_current = doc_topic_distributions[i]
        n_list = []
        for j in range(i-L, i-1):
            n_list.append(np.var(doc_topic_distributions[j])*Diff(d_current, doc_topic_distributions[j]))
        weight = 0*np.ones([topic_num])
        weight[0] = 1
        n_outlier = np.var(weight)*epsilon
        p_list = []
        for n in n_list:
            p_list.append(n / (n_outlier + np.array(n_list).sum()))
        p_list.append(n_outlier / (n_outlier + np.array(n_list).sum()))
        p_list = np.array(p_list)
        eta_list[i] = p_list.copy()
    
# topic assignment for a word
def get_a_topic(doc_topic_distribution):
    topics = np.random.multinomial(1, doc_topic_distribution)
    topic = -1
    for i in range(0, len(topics)):
        if topics[i] > 0:
            topic = i
            break
    return topic

base = 0.000001
delta = 0.00001
epsilon = abs(base**(0.25+delta))
#print(epsilon)


# intialization for all distributions
def initialize_distributions():
    global doc_topic_distributions, topic_word_distributions, topic_word_distribution
    doc_topic_distributions.clear()
    topic_word_distributions.clear()
    topic_word_distribution.clear()
    for i in range(0, docs_num):
        doc_topic_distributions.append(1./topic_num*np.ones([topic_num]))
        topics_pdf = [] 
        for j in range(0, topic_num):
            topics_pdf.append(1./words_num*np.ones([words_num]))
        topic_word_distributions.append(topics_pdf)
    for i in range(0, topic_num):
        topic_word_distribution.append(1./words_num*np.ones([words_num]))
    return


# initialize documents and their corressponding topics
def initial_docs_list():
    global data, docs_list
    docs_list.clear()
    for doc in data:
         docs_list.append(np.ones([len(doc), 2], dtype = np.uint8))
    return

def initialize_values_docs_list():
    global docs_list
    for d in range(0, len(data)):
        for w in range(0, len(data[d])):
           docs_list[d][w] = [word_index[data[d][w]], get_a_topic(doc_topic_distributions[d])]
    return


# compute topic assignments for each word in each document
def compute_doc_topic():
    global doc_topic
    doc_topic = np.array(doc_topic)
    doc_topic = 0*doc_topic
    for i in range(len(docs_list)):
        for j in range(0, len(docs_list[i])):
            doc_topic[i][docs_list[i][j][1]] += 1
end = 1
# compute topic assignment for document d
def compute_doc_topic_doc(d):
    global doc_topic
    doc_topic[d] = np.array(doc_topic[d])
    doc_topic[d] = 0*doc_topic[d]
#    print(doc_topic[d])
    for j in range(0, len(docs_list[d])):
        doc_topic[d][docs_list[d][j][1]] += 1

# compute topic-word distribuion
def compute_topic_word():
    global topic_word
    topic_word = np.array(topic_word)
    topic_word = 0*topic_word
    for i in range(len(docs_list)):
        for j in range(0, len(docs_list[i])):
            topic_word[docs_list[i][j][1]][docs_list[i][j][0]] += 1
    return

# compute topic word distribution list
def compute_topic_word_list_doc(d):  
    global docs_list
    topic_word_list[d] = np.array(topic_word_list[d])
    topic_word_list[d] = 0*topic_word_list[d]
    for i in range(len(docs_list)):
        for j in range(0, len(docs_list[i])):
            topic_word_list[d][docs_list[i][j][1]][docs_list[i][j][0]] += 1
    return


# get the number of word w assigned by topic k in document d
def get_n_d_k(d, w, k):
    n_d_k = 0
    for i in range(0, len(docs_list[d])):
        if(i != w and docs_list[d][i][1]- k == 0):
            n_d_k += 1
    return n_d_k

# get the number of word w assigned by topic k in all documents except the current one
def get_n_w_k(d, w, k):
    n_w_k = 0
    if(docs_list[d][w][1] - k == 0):
        n_w_k = topic_word_list[d][k][docs_list[d][w][0]] - 1
    else:
        n_w_k = topic_word_list[d][k][docs_list[d][w][0]]
    return n_w_k

# get the number of words assigned by topic k in document d
def get_total_n_k(d, w, k):
    total_n_k = np.sum(topic_word_list[d][k])
    if(docs_list[d][w][1] - k == 0):
        total_n_k = total_n_k - 1
    return total_n_k

# recompute topic distributions of word w
def recompute_w_topic_distribution_for_fSOT(d, w, td_list, eta):
    new_topic_distribution = np.ones([topic_num])
    num_list = np.ones([topic_num])

    for topic in range(0, topic_num):
        n_d_k = get_n_d_k(d, w, topic)
        n_w_k = get_n_w_k(d, w, topic)
        total_n_k = get_total_n_k(d, w, topic)
        former = 0
        scope = 0
        for t in td_list:
            former += eta[scope]*(n_d_k + alpha*t[topic])
            scope += 1
        former += eta[scope]*(n_d_k + alpha)
        p_d_w_k = former*(n_w_k + beta)/(total_n_k + words_num*beta)
        num_list[topic] = p_d_w_k
        new_topic_distribution[topic] = p_d_w_k
    new_topic_distribution = new_topic_distribution / np.sum(new_topic_distribution) 
    return new_topic_distribution

# get the list of topic distribuions of the set of previous documents
def ge_dt_list(d):
    dt_list = []
    if(d-L < 0):
        strt = 0
    else:
        strt = d-L
    for i in range(strt, d - 1):
        dt_list.append(doc_topic_distributions[i])
    return dt_list

# iteration of gibbs sampling
def gibbs_sampling():
    global doc_topic_distributions, eta_list, st, ed, total_time
    st = 0
    ed = 0
    total_time = 0
    
    for d in range(0, len(docs_list)):
        st = time.time()
        for w in range(0, len(docs_list[d])):
            dt_list = ge_dt_list(d)
            new_pdf = recompute_w_topic_distribution_for_fSOT(d, w, dt_list, eta_list[d])               
            new_topic = get_a_topic(new_pdf)
            docs_list[d][w][1] = new_topic
        ed = time.time()
        total_time += ed - st
        compute_doc_topic_doc(d)
        compute_topic_word_list_doc(d)
        recompute_distributions_doc(d)

# recompute topic distribution for document d
def recompute_distributions_doc(d):
    doc_topic_distributions[d] = (doc_topic[d] + alpha) / (np.sum(doc_topic[d]) + len(doc_topic[d]) * alpha)
    for topic in range(0, len(topic_word)):
        topic_word_distributions[d][topic] = (topic_word_list[d][topic] + beta) / (np.sum(topic_word_list[d][topic]) + len(topic_word_list[d][topic]) * beta)
   
# recompute all distribuiotns      
def recompute_distributions():
    for d in range(0, len(doc_topic)):
        doc_topic_distributions[d] = (doc_topic[d] + alpha) / (np.sum(doc_topic[d]) + len(doc_topic[d]) * alpha)
        for topic in range(0, len(topic_word)):
            topic_word_distributions[d][topic] = (topic_word_list[d][topic] + beta) / (np.sum(topic_word_list[d][topic]) + len(topic_word_list[d][topic]) * beta)
    for topic in range(0, len(topic_word)):
        topic_word_distribution[topic] = (topic_word[topic] + beta) / (np.sum(topic_word[topic]) + len(topic_word[topic]) * beta)

def compare_distributions(dlist1, dlist2):
    result = np.ones([len(dlist2)])  
    if(len(dlist1) != len(dlist2)):
        return 
    for i in range(0, len(dlist1)):
        result[i] = symmetricalKL(dlist1[i], dlist2[i])
    return result

def compute_perplexities():
    global doc_topic_distributions, topic_word_distribution, docs_list
    total = 0
    total_num = 0
    for d in range(0, len(docs_list)):
        for v in range(0, len(docs_list[d])):
            total_topics = 0
            for k in range(0, len(topic_word_distribution)):
                theta_d_k = doc_topic_distributions[d][k]
                if(model_name != "LDA"):
                    phi_d_k_v = topic_word_distributions[d][k][docs_list[d][v][0]]
                else:
                    phi_d_k_v = topic_word_distribution[k][docs_list[d][v][0]]
                total_topics += theta_d_k*phi_d_k_v
            total_num += 1.0
            total += (-1)*math.log(total_topics)
    
    return math.exp(total / total_num) 

def parameter_estimation():
    per_list.clear()
    res_list1.clear()
    res_list2.clear()
    print(model_name)
    for i in range(0, iteration_num):
        compute_eta_list(L)
        d1 = doc_topic_distributions.copy()
        d2 = topic_word_distribution.copy()
        gibbs_sampling()
        print(model_name + "_Iteration" , i, " time:  ", total_time)
        recompute_distributions()
        if(model_name == "LDA"):
            compute_doc_topic()
            compute_topic_word()     
        res_list1.append(np.mean(compare_distributions(d1, doc_topic_distributions)))
        res_list2.append(np.mean(compare_distributions(d2, topic_word_distribution)))
        per_list.append(compute_perplexities())
    return
        
def save_result(path):
        # SAVE
    if not os.path.exists(path):
        os.makedirs(path)
    LDA_docs_list = np.array(docs_list) 
    LDA_doc_topic_distributions = np.array(doc_topic_distributions)
    LDA_topic_word_distributions = np.array(topic_word_distributions)
    LDA_topic_word_distribution = np.array(topic_word_distribution)
    np.save(path + str(model_name)+"docs_list"+str(topic_num)+".npy", LDA_docs_list)
    np.save(path + str(model_name)+"doc_topic_distributions_"+str(topic_num)+".npy", LDA_doc_topic_distributions)
    np.save(path + str(model_name)+"topic_word_distributions_"+str(topic_num)+".npy", LDA_topic_word_distributions)
    np.save(path + str(model_name)+"topic_word_distribution_"+str(topic_num)+".npy", LDA_topic_word_distribution)
    LDA_per_list = np.array(per_list)
    np.save(path + str(model_name)+"per_list"+str(topic_num)+".npy", LDA_per_list)
    LDA_eta_list = np.array(eta_list)
    np.save(path + str(model_name)+"eta_list"+str(topic_num)+".npy", LDA_eta_list)
    LDA_res_list1 = np.array(res_list1)
    np.save(path + str(model_name)+"res_list1"+str(topic_num)+".npy", LDA_res_list1)
    LDA_res_list2 = np.array(res_list2)
    np.save(path + str(model_name)+"res_list2"+str(topic_num)+".npy", LDA_res_list2)

def initialize():
    global topic_word, doc_topic, topic_word_list
    topic_word = 0*np.ones([topic_num, words_num])
    doc_topic = 0*np.ones([docs_num, topic_num])
    topic_word_list = 0*np.ones([docs_num, topic_num, words_num])
    initialize_distributions()
    initial_docs_list()
    initialize_values_docs_list()
    compute_doc_topic()
    compute_topic_word()
    for i in range(0, docs_num):
        compute_topic_word_list_doc(i)
    return

def run(t_data, start, end_iter, iterations, save_p, clip, lambda_, pL, palpha, pbeta, delta):  
    global topic_num, iteration_num, data_clip, epsilon, data, docs_num, topic_num, words_num, L, eta_list, alpha, beta
    data.clear()
    eta_list.clear()
    data=t_data
    alpha = palpha
    beta = pbeta
    save_path = save_p
    data_clip = clip
    topic_num = start
    iteration_num = iterations
    epsilon = abs(base**(lambda_+delta))
    L = pL 
    docs_num = data_clip
    eta_list =  np.ones([docs_num, L])
    
    create_dictionary(data) 
    docs_num = len(data)
    topic_num = start
    words_num = len(word_index)
    for i in range(0, end_iter):
        initialize()
        parameter_estimation()
        save_result(save_path)
        topic_num += 2
        np.save("f_SOT_runtime_"+str(data_clip)+".npy", total_time)
    return 
 

