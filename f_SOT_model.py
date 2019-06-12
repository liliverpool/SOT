# -*- coding: utf-8 -*-
"""
Created on Thu March 12 17:55:50 2019

@author: slab
"""

import pymc3 as pm
import numpy as np
import math
import time
import os
import lda
import lda.datasets

model_name = "f_SOT"



start = 9
end = 1
L = 15
data_clip = 500
stop_file = open('stopwords2.txt', 'r')
readtext = stop_file.read()
stop_list = readtext.split('\n')
data = []
word_index = dict()
index_word = dict()
docs_num = 1
topic_num = start
words_num = 1
alpha = 0.5
beta = 0.5
eta = 1
gamma = 1
delta = 0.00001
base = 0.000001
epsilon = abs(base**(0.25+delta))
decay = 0.05
iteration_num = 50
topic_word = 0*np.ones([1, 1])
topic_word_list = 0*np.ones([1, 1, 1])
doc_topic = 0*np.ones([1, 1])
docs_list = []
doc_topic_distributions = []
topic_word_distributions = []
topic_word_distribution = []
perplexities = []


eta_list =  []
gamma_list = []

res_list1 = []
res_list2 = []
per_list = []
st= 0
ed= 0
total_time = 0

def create_dictionary(data):
    global word_index, index_word
    for doc in data:
        for w in doc:
            if w not in word_index:
                word_index[w] = len(word_index)
    index_word = dict(zip(word_index.values(), word_index.keys()))
    
def compute_mean_topics_no_decay(beafore_d):
    global doc_topic_distributions, decay
    d_sum = 0*np.ones([topic_num])
    n_sum = 0
    if(beafore_d > len(doc_topic_distributions)):
        beafore_d = len(doc_topic_distributions)
    if(beafore_d < 2):
        return doc_topic_distributions[beafore_d]
    for i in range(0, beafore_d):
        d_sum += doc_topic_distributions[i]
        n_sum += 1
    return d_sum/n_sum

def compute_mean_topics_SOT(beafore_d, L):
    global doc_topic_distributions, decay
    d_sum = 0*np.ones([topic_num])
    n_sum = 0
    if(beafore_d > len(doc_topic_distributions)):
        beafore_d = len(doc_topic_distributions)
    if(beafore_d < 2):
        return doc_topic_distributions[beafore_d]
    for i in range(L, beafore_d):
        d_sum += doc_topic_distributions[i]
        n_sum += 1
    return d_sum/n_sum

def compute_mean_words_no_decay(beafore_d, k):
    global topic_word_distributions, decay
    d_sum = 0*np.ones([words_num])
    n_sum = 0
    if(beafore_d > len(topic_word_distributions)):
        beafore_d = len(topic_word_distributions)
    if(beafore_d < 2):
        return topic_word_distributions[beafore_d][k]
    for i in range(0, beafore_d):
        d_sum += doc_topic_distributions[i][k]
        n_sum += 1
    return d_sum/n_sum

def rebuild_reuters_data(data, voc):
    new_data = []
    for doc in data:
        doc_words = []
        for word_index in range(0, len(doc)):
            word_num = doc[word_index]
            for i in range(0, word_num):
                doc_words.append(voc[word_index])
        new_data.append(doc_words)
    return new_data

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

def compute_mean_topics(beafore_d):
    global doc_topic_distributions, decay
    d_sum = 0*np.ones([topic_num])
    n_sum = 0
    if(beafore_d > len(doc_topic_distributions)):
        beafore_d = len(doc_topic_distributions)
    if(beafore_d < 1):
        return doc_topic_distributions[beafore_d]

    for i in range(0, beafore_d):
        d_sum += 2**((-1)*decay*(beafore_d-i))*doc_topic_distributions[i]
        n_sum += 2**((-1)*decay*(beafore_d-i))
    return d_sum/n_sum



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

def compute_mean_words(beafore_d, k):
#    print("asdasd", beafore_d, k)
    global topic_word_distributions, decay
    d_sum = 0*np.ones([words_num])
    n_sum = 0
    if(beafore_d > len(topic_word_distributions)):
        beafore_d = len(topic_word_distributions)
    if(beafore_d < 2):
#        print("asdasd", beafore_d, k)
        return topic_word_distributions[beafore_d][k]
    for i in range(0, beafore_d):
        d_sum += 2**((-1)*decay*(beafore_d-i))*doc_topic_distributions[i][k]
        n_sum += 2**((-1)*decay*(beafore_d-i))
    return d_sum/n_sum

def compute_gamma_list():
    global topic_word_distributions, epsilon, gamma_list
    gamma_list.clear()
    print(topic_num)
    for i in range(0, len(topic_word_distributions)):
        gamma_topics = []

        for k in range(0, topic_num):
            if(i < 1):
                gamma_topics.append(np.array([0, 0, 1]))
            else:
                d_mean = compute_mean_words(i, k)
                d_previuos = topic_word_distributions[i-1][k]
                d_current = topic_word_distributions[i][k]
                n_0 = np.var(d_mean)*Diff(d_current, d_mean) 
                n_1 = np.var(d_previuos)*Diff(d_current, d_previuos)  
                weight = 0*np.ones([words_num])
                weight[0] = 1
                n_2 = np.var(weight)*epsilon
                p_0 = n_0 / (n_0 + n_1 + n_2)
                p_1 = n_1 / (n_0 + n_1 + n_2)
                p_2 = n_2 / (n_0 + n_1 + n_2)
                gamma_topics.append(np.array([p_0, p_1, p_2]))
        gamma_list.append(gamma_topics)
    

def get_a_topic(doc_topic_distribution):
    topics = np.random.multinomial(1, doc_topic_distribution)
    topic = -1
    for i in range(0, len(topics)):
        if topics[i] > 0:
            topic = i
            break
    return topic

def get_a_topic_old(doc_topic_distribution):
    z = pm.distributions.multivariate.Multinomial.dist(1, doc_topic_distribution)
    topics = z.random()
    topic = -1
    for i in range(0, len(topics)):
        if topics[i] > 0:
            topic = i
            break
    return topic

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

def compute_doc_topic():
    global doc_topic
    doc_topic = np.array(doc_topic)
    doc_topic = 0*doc_topic
    for i in range(len(docs_list)):
        for j in range(0, len(docs_list[i])):
            doc_topic[i][docs_list[i][j][1]] += 1

def compute_doc_topic_doc(d):
    global doc_topic
    doc_topic[d] = np.array(doc_topic[d])
    doc_topic[d] = 0*doc_topic[d]
#    print(doc_topic[d])
    for j in range(0, len(docs_list[d])):
        doc_topic[d][docs_list[d][j][1]] += 1

def compute_topic_word():
    global topic_word
    topic_word = np.array(topic_word)
    topic_word = 0*topic_word
    for i in range(len(docs_list)):
        for j in range(0, len(docs_list[i])):
            topic_word[docs_list[i][j][1]][docs_list[i][j][0]] += 1
    return

def compute_topic_word_list_doc(d):  
    global docs_list
    topic_word_list[d] = np.array(topic_word_list[d])
    topic_word_list[d] = 0*topic_word_list[d]
    for i in range(len(docs_list)):
        for j in range(0, len(docs_list[i])):
            topic_word_list[d][docs_list[i][j][1]][docs_list[i][j][0]] += 1
    return

def get_n_d_k(d, w, k):
    n_d_k = 0
    for i in range(0, len(docs_list[d])):
        if(i != w and docs_list[d][i][1]- k == 0):
            n_d_k += 1
    return n_d_k

def get_n_w_k_2(d, w, k):
    n_w_k = 0
    if(docs_list[d][w][1] - k == 0):
        n_w_k = topic_word_list[d][k][docs_list[d][w][0]] - 1
    else:
        n_w_k = topic_word_list[d][k][docs_list[d][w][0]]
    return n_w_k

def get_total_n_k_2(d, w, k):
    total_n_k = np.sum(topic_word_list[d][k])
    if(docs_list[d][w][1] - k == 0):
        total_n_k = total_n_k - 1
    return total_n_k

def get_n_w_k(d, w, k):
    n_w_k = 0
    if(docs_list[d][w][1] - k == 0):
        n_w_k = topic_word[k][docs_list[d][w][0]] - 1
    else:
        n_w_k = topic_word[k][docs_list[d][w][0]]
    return n_w_k

def get_total_n_k(d, w, k):
    total_n_k = np.sum(topic_word[k])
    if(docs_list[d][w][1] - k == 0):
        total_n_k = total_n_k - 1
    return total_n_k

def recompute_w_topic_distribution(d, w):
    new_topic_distribution = np.ones([topic_num])
    num_list = np.ones([topic_num])
    for topic in range(0, topic_num):
        n_d_k = get_n_d_k(d, w, topic)
        n_w_k = get_n_w_k(d, w, topic)
        total_n_k = get_total_n_k(d, w, topic)
        p_d_w_k = (n_d_k + alpha)*(n_w_k + beta)/(total_n_k + words_num*beta)
        num_list[topic] = p_d_w_k
        new_topic_distribution[topic] = p_d_w_k
    new_topic_distribution = new_topic_distribution / np.sum(new_topic_distribution)        
    return new_topic_distribution

def recompute_w_topic_distribution_for_SOT_2(d, w, td_list, dw_means, dw_previuos, eta, gamma, total_beta_mean_list, total_beta_previous_list):
    new_topic_distribution = np.ones([topic_num])
    num_list = np.ones([topic_num])

    for topic in range(0, topic_num):
        n_d_k = get_n_d_k(d, w, topic)
        n_w_k = get_n_w_k_2(d, w, topic)
        total_n_k = get_total_n_k_2(d, w, topic)
        total_beta_mean = total_beta_mean_list[topic]
        total_beta_previuos = total_beta_previous_list[topic]
        former = 0
        scope = 0
        for t in td_list:
            former += eta[scope]*(n_d_k + alpha*t[topic])
            scope += 1
        former += eta[scope]*(n_d_k + alpha)
        p_d_w_k = former*(gamma[topic][0]*(n_w_k + beta*dw_means[topic][docs_list[d][w][0]])/(total_n_k + total_beta_mean) \
                    + gamma[topic][1]*(n_w_k + beta*dw_previuos[topic][docs_list[d][w][0]])/(total_n_k + total_beta_previuos) \
                    + gamma[topic][2]*(n_w_k + beta)/(total_n_k + words_num*beta))
        num_list[topic] = p_d_w_k
        new_topic_distribution[topic] = p_d_w_k
    new_topic_distribution = new_topic_distribution / np.sum(new_topic_distribution) 
    return new_topic_distribution

def recompute_w_topic_distribution_for_TMT_2(d, w, td_mean, td_previuos, dw_means, dw_previuos, eta, gamma, total_beta_mean_list, total_beta_previous_list):
    new_topic_distribution = np.ones([topic_num])
    num_list = np.ones([topic_num])

    for topic in range(0, topic_num):
        n_d_k = get_n_d_k(d, w, topic)
        n_w_k = get_n_w_k_2(d, w, topic)
        total_n_k = get_total_n_k_2(d, w, topic)
        total_beta_mean = total_beta_mean_list[topic]
        total_beta_previuos = total_beta_previous_list[topic]
        p_d_w_k = (eta[0]*(n_d_k + alpha*td_mean[topic])+ eta[1]*(n_d_k + alpha*td_previuos[topic])+ eta[2]*(n_d_k + alpha)) \
                    *(gamma[topic][0]*(n_w_k + beta*dw_means[topic][docs_list[d][w][0]])/(total_n_k + total_beta_mean) \
                    + gamma[topic][1]*(n_w_k + beta*dw_previuos[topic][docs_list[d][w][0]])/(total_n_k + total_beta_previuos) \
                    + gamma[topic][2]*(n_w_k + beta)/(total_n_k + words_num*beta))
        num_list[topic] = p_d_w_k
        new_topic_distribution[topic] = p_d_w_k
    new_topic_distribution = new_topic_distribution / np.sum(new_topic_distribution)   
    return new_topic_distribution

def recompute_w_topic_distribution_for_sDCT_2(d, w, td_previuos, dw_previuos, total_beta_previous_list):
    new_topic_distribution = np.ones([topic_num])
    num_list = np.ones([topic_num])
    for topic in range(0, topic_num):
        n_d_k = get_n_d_k(d, w, topic)
        n_w_k = get_n_w_k_2(d, w, topic)
        total_n_k = get_total_n_k_2(d, w, topic)
#        total_beta_mean = compute_total_beta(dw_means, topic, gamma)
        total_beta_previuos = total_beta_previous_list[topic]
        p_d_w_k = (n_d_k + alpha*td_previuos[topic])*(n_w_k + beta*dw_previuos[topic][docs_list[d][w][0]])/(total_n_k + total_beta_previuos)
#        print(p_d_w_k)
        num_list[topic] = p_d_w_k
        new_topic_distribution[topic] = p_d_w_k
    new_topic_distribution = new_topic_distribution / np.sum(new_topic_distribution)     
    return new_topic_distribution

def recompute_w_topic_distribution_for_lDCT_2(d, w, td_mean, dw_mean, total_beta_mean_list):
    new_topic_distribution = np.ones([topic_num])
    num_list = np.ones([topic_num])
    for topic in range(0, topic_num):
        n_d_k = get_n_d_k(d, w, topic)
        n_w_k = get_n_w_k_2(d, w, topic)
        total_n_k = get_total_n_k_2(d, w, topic)
        total_beta_mean = total_beta_mean_list[topic]
        p_d_w_k = (n_d_k + alpha*td_mean[topic])*(n_w_k + beta*dw_mean[topic][docs_list[d][w][0]])/(total_n_k + total_beta_mean)

        num_list[topic] = p_d_w_k
        new_topic_distribution[topic] = p_d_w_k
    new_topic_distribution = new_topic_distribution / np.sum(new_topic_distribution)        
    return new_topic_distribution

def recompute_w_topic_distribution_for_sLDA_2(d, w, td_previuos, dw_previuos, eta, gamma, total_beta_previous_list):
    new_topic_distribution = np.ones([topic_num])
    num_list = np.ones([topic_num])
    for topic in range(0, topic_num):
        n_d_k = get_n_d_k(d, w, topic)
        n_w_k = get_n_w_k_2(d, w, topic)
        total_n_k = get_total_n_k_2(d, w, topic)
        total_beta_previuos = total_beta_previous_list[topic]
        p_d_w_k = (n_d_k + alpha+eta[1]*td_previuos[topic])*(n_w_k + beta+gamma[topic][1]*dw_previuos[topic][docs_list[d][w][0]])/(total_n_k + total_beta_previuos)

        num_list[topic] = p_d_w_k
        new_topic_distribution[topic] = p_d_w_k
    new_topic_distribution = new_topic_distribution / np.sum(new_topic_distribution)        
    return new_topic_distribution

def recompute_w_topic_distribution_for_TMT(d, w, td_mean, td_previuos, eta):
    new_topic_distribution = np.ones([topic_num])
    num_list = np.ones([topic_num])
    for topic in range(0, topic_num):
        n_d_k = get_n_d_k(d, w, topic)
        n_w_k = get_n_w_k(d, w, topic)
        total_n_k = get_total_n_k(d, w, topic)
        p_d_w_k = eta[0]*(n_d_k + alpha*td_mean[topic])*(n_w_k + beta)/(total_n_k + words_num*beta) \
                        + eta[1]*(n_d_k + alpha*td_previuos[topic])*(n_w_k + beta)/(total_n_k + words_num*beta) \
                        + eta[2]*(n_d_k + alpha)*(n_w_k + beta)/(total_n_k + words_num*beta)

        num_list[topic] = p_d_w_k
        new_topic_distribution[topic] = p_d_w_k
    new_topic_distribution = new_topic_distribution / np.sum(new_topic_distribution)        
    return new_topic_distribution

def recompute_w_topic_distribution_for_SOT(d, w, td_mean, td_previuos, eta):
    new_topic_distribution = np.ones([topic_num])
    num_list = np.ones([topic_num])
    for topic in range(0, topic_num):
        n_d_k = get_n_d_k(d, w, topic)
        n_w_k = get_n_w_k(d, w, topic)
        total_n_k = get_total_n_k(d, w, topic)
        p_d_w_k = eta[0]*(n_d_k + alpha*td_mean[topic])*(n_w_k + beta)/(total_n_k + words_num*beta) \
                        + eta[1]*(n_d_k + alpha*td_previuos[topic])*(n_w_k + beta)/(total_n_k + words_num*beta) \
                        + eta[2]*(n_d_k + alpha)*(n_w_k + beta)/(total_n_k + words_num*beta)

        num_list[topic] = p_d_w_k
        new_topic_distribution[topic] = p_d_w_k
    new_topic_distribution = new_topic_distribution / np.sum(new_topic_distribution)        
    return new_topic_distribution

def recompute_w_topic_distribution_for_sDCT(d, w, td_previuos):
    new_topic_distribution = np.ones([topic_num])
    num_list = np.ones([topic_num])
    for topic in range(0, topic_num):
        n_d_k = get_n_d_k(d, w, topic)
        n_w_k = get_n_w_k(d, w, topic)
        total_n_k = get_total_n_k(d, w, topic)
        p_d_w_k = (n_d_k + alpha*td_previuos[topic])*(n_w_k + beta)/(total_n_k + words_num*beta)
#        print(p_d_w_k)
        num_list[topic] = p_d_w_k
        new_topic_distribution[topic] = p_d_w_k
    new_topic_distribution = new_topic_distribution / np.sum(new_topic_distribution)        
    return new_topic_distribution

def recompute_w_topic_distribution_for_lDCT(d, w, td_mean):
    new_topic_distribution = np.ones([topic_num])
    num_list = np.ones([topic_num])
    for topic in range(0, topic_num):
        n_d_k = get_n_d_k(d, w, topic)
        n_w_k = get_n_w_k(d, w, topic)
        total_n_k = get_total_n_k(d, w, topic)
        p_d_w_k = (n_d_k + alpha*td_mean[topic])*(n_w_k + beta)/(total_n_k + words_num*beta)
#        print(p_d_w_k)
        num_list[topic] = p_d_w_k
        new_topic_distribution[topic] = p_d_w_k
    new_topic_distribution = new_topic_distribution / np.sum(new_topic_distribution)        
    return new_topic_distribution



def compute_total_beta(dw, topic, gamma = 1):
    total = 0
    for i in range(0, words_num):
        if(model_name != "sLDA"):
            total += gamma*beta*dw[topic][i]
        else:
            total += beta+gamma*dw[topic][i]
    return total

def get_beta_total(dw, gamma):
    total_beta_list = []
    for i in range(0, topic_num):
        if(gamma == -1):
            total_beta_list.append(compute_total_beta(dw, i))
        else:
            total_beta_list.append(compute_total_beta(dw, i, gamma[i][1]))
    return total_beta_list

def ge_dt_list(d):
    dt_list = []
    if(d-L < 0):
        strt = 0
    else:
        strt = d-L
    for i in range(strt, d - 1):
        dt_list.append(doc_topic_distributions[i])
    return dt_list
    
def gibbs_sampling():
    global doc_topic_distributions, eta_list, gamma_list, st, ed, total_time
    st = 0
    ed = 0
    total_time = 0
    
    for d in range(0, len(docs_list)):
        dw_means = []
        if(model_name == "SOT"):
            td_mean = compute_mean_topics(d)
            for k in range(0, topic_num):
                dw_means.append(compute_mean_words(d, k))
        else:
            td_mean = compute_mean_topics_no_decay(d)
            for k in range(0, topic_num):
                dw_means.append(compute_mean_words_no_decay(d, k))
        
        if(d < 1):
            td_previuos = doc_topic_distributions[d]
            dw_previous = topic_word_distributions[d]
        else:
            td_previuos = doc_topic_distributions[d-1]
            dw_previous = topic_word_distributions[d-1]
        
        if(model_name == "sLDA"):
            total_beta_previous_list = get_beta_total(dw_previous, gamma_list[d])
        else:
            total_beta_previous_list = get_beta_total(dw_previous, -1)
            total_beta_mean_list = get_beta_total(dw_means, -1)
        
        st = time.time()
        for w in range(0, len(docs_list[d])):
            
            if(model_name == "f_SOT"):
                dt_list = ge_dt_list(d)
                new_pdf = recompute_w_topic_distribution_for_SOT_2(d, w, dt_list, dw_means, dw_previous, eta_list[d], gamma_list[d], total_beta_mean_list, total_beta_previous_list)
            elif(model_name == "sDCT"):
                new_pdf = recompute_w_topic_distribution_for_sDCT_2(d, w, td_previuos, dw_previous, total_beta_previous_list)
            elif(model_name == "lDCT"):
                new_pdf = recompute_w_topic_distribution_for_lDCT_2(d, w, td_mean, dw_means, total_beta_mean_list)
            elif(model_name == "sLDA"):
                new_pdf = recompute_w_topic_distribution_for_sLDA_2(d, w, td_previuos, dw_previous, eta_list[d], gamma_list[d], total_beta_previous_list)
            else:
                new_pdf = recompute_w_topic_distribution(d, w)
            new_topic = get_a_topic(new_pdf)
            docs_list[d][w][1] = new_topic
        ed = time.time()
        total_time += ed - st
        if(model_name != "LDA"):
            compute_doc_topic_doc(d)
            compute_topic_word_list_doc(d)
            recompute_distributions_doc(d)
            

def TMT_gibbs_sampling():
    for d in range(0, len(docs_list)):
        for w in range(0, len(docs_list[d])):
            new_pdf = recompute_w_topic_distribution(d, w)
            new_topic = get_a_topic(new_pdf)
            docs_list[d][w][1] = new_topic

def recompute_distributions_doc(d):
    doc_topic_distributions[d] = (doc_topic[d] + alpha) / (np.sum(doc_topic[d]) + len(doc_topic[d]) * alpha)
    for topic in range(0, len(topic_word)):
        topic_word_distributions[d][topic] = (topic_word_list[d][topic] + beta) / (np.sum(topic_word_list[d][topic]) + len(topic_word_list[d][topic]) * beta)
           
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
        compute_gamma_list()
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
    LDA_gamma_list = np.array(gamma_list)
    np.save(path + str(model_name)+"gamma_list"+str(topic_num)+".npy", LDA_gamma_list)
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
 












