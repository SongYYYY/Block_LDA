import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt 
import random 
import networkx as nx 
import itertools 
from scipy.special import digamma
import seaborn as sns 

def get_top_docs(omega,labels,top=10,with_value=True):
    results=[]
    for i in range(omega.shape[0]):
        sorted_idx=np.argsort(-omega[i,:])[:top]
        top_labels=list(labels[sorted_idx])
        top_values=list(omega[i,sorted_idx])
        if with_value:
            results.append(list(zip(top_labels,top_values)))
        else:
            results.append(top_labels)
            
    return results 

def get_top_tokens(phi,tokens,top=10,with_value=True):
    results=[]
    for i in range(phi.shape[0]):
        sorted_idx=np.argsort(-phi[i,:])[:top]
        sorted_tokens=list(tokens[sorted_idx])
        sorted_values=list(phi[i,sorted_idx])
        if with_value:
            results.append(list(zip(sorted_tokens,sorted_values))) 
        else:
            results.append(sorted_tokens)
    
    return results 


def get_normalized_theta(theta,method='row'):
    if method=='row':
        norm=theta.sum(axis=1)[:,np.newaxis]
    elif method=='col':
        norm=theta.sum(axis=0)[np.newaxis,:]
    result=theta/norm
    
    return result

def draw_image_matrix(omega,labels,figsize=(10,10)):
    label_ids_dict={}
    for i,l in enumerate(labels):
        if l in label_ids_dict:
            label_ids_dict[l].append(i)
        else:
            label_ids_dict[l]=[i]
            
    image_mat=np.zeros_like(omega.T)
    row=0
    for k in label_ids_dict.keys():
        for idx in label_ids_dict[k]:
            image_mat[row,:]=omega[:,idx]
            row+=1 
            
    fig,ax=plt.subplots(figsize=figsize) 
    sns.heatmap(image_mat,robust=True,ax=ax)
    
    return 

def get_sub_input(classes,G,labels,texts):
    '''
    args:
    classes: list of strs
    G: original graph (DiGraph)
    labels: original labels
    texts: original BOW texts
    
    return:
    G_sub_idx: subgraph from G containing nodes in the specified classes and edges between them. 
        Isolates are removed.
    labels_sub: labels consistent with G_sub_idx
    texts_sub: BOW texts consistent with G_sub_idx 
    sub_idx_dict: the map of (original idx: new_idx) 
    '''
    sub_idx_list=[]
    for i,l in enumerate(labels):
        if l in classes:
            sub_idx_list.append(i)
            
    G_sub=nx.DiGraph(nx.subgraph(G,sub_idx_list)) 
    G_sub.remove_nodes_from(list(nx.isolates(G_sub)))
    
    sub_idx_dict={v:k for k,v in enumerate(list(G_sub.nodes))}
    sub_idx_list=list(sub_idx_dict.keys())
    
    G_sub_idx=nx.DiGraph()
    edges=[(sub_idx_dict[i],sub_idx_dict[j]) for i,j in G_sub.edges]
    G_sub_idx.add_edges_from(edges)
    
    labels_sub=labels[sub_idx_list]
    texts_sub=texts[sub_idx_list] 
    
    return G_sub_idx,labels_sub,texts_sub,sub_idx_dict 


def get_cluster_results(omega):
    result=[]
    for i in range(omega.shape[1]):
        result.append(np.argmax(omega[:,i]))
    return result 


def get_train_graph(G,removed_edges_ratio=0.2,seed=1):
    G=G.copy() 
    removed_edges_ratio=0.2  
    removed_edges_number=int(len(G.edges)*removed_edges_ratio) 
    rs=random.getstate()
    random.seed(seed)
    removed_edges=random.sample(list(G.edges),removed_edges_number) 
    random.setstate(rs)
    G.remove_edges_from(removed_edges)
    
    return G,removed_edges 

def predict_links(G,K,theta,omega,num_pred_edges):
    edges_to_be_pred=list(set(np.ndindex(G.number_of_nodes(),G.number_of_nodes()))-set(G.edges))
    edges_proba=[]
    for i,j in edges_to_be_pred:
        p=(theta.flatten()*np.repeat(omega[:,i],K)*np.tile(omega[:,j],K)).sum()
        edges_proba.append((i,j,p))
    sorted_edges_proba=sorted(edges_proba,key=lambda x:x[2],reverse=True)
    return [(i,j) for i,j,k in sorted_edges_proba[:num_pred_edges]]


def pred_links_acc(pred_links,true_links):
    correct=len(set.intersection(set(pred_links),set(true_links))) 
    acc=correct/len(pred_links)
    return correct,acc 


