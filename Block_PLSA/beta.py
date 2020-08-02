import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt 
import random 
import networkx as nx 
import itertools 
import pickle 
from scipy.special import digamma
import timeit

# utils 
def index2ij(index,K):
    row=int(index/K) 
    col=index%K
        
    return row,col

def get_top_tokens(phi,tokens,top=10):
    results=[]
    for i in range(phi.shape[0]):
        results.append(tokens[np.argsort(-phi[i,:])[:top]])
    return results 

def get_top_docs(omega,labels,top=10):
    results=[]
    for i in range(omega.shape[0]):
        results.append(labels[np.argsort(-omega[i,:])[:top]])
    return results 



# class 
class Block_PLSA:
    def __init__(self,G,texts):
        self.texts=texts
        self.G=G

        return 
    

 
    # input_transformation
    def transfer(self,G,texts):
        # transfer input to observed variables
        # PLSA part
        ii,jj=np.nonzero(texts)
        WS=np.repeat(jj,texts[ii,jj])
        DS=np.repeat(ii,texts[ii,jj])
        # blockmodel part
        SS=[]
        RS=[]
        for e in G.edges:
            SS.append(e[0])
            RS.append(e[1])
        SS=np.array(SS,dtype=np.int)
        RS=np.array(RS,dtype=np.int)
    
        return WS,DS,SS,RS

    # initialize
    def initialize_helpers(self,WS,DS,SS,RS):
        # help utils
        D_ids={}
        for idx,d in enumerate(DS):
            if d in D_ids:
                D_ids[d].append(idx)
            else:
                D_ids[d]=[idx]
        
        S_ids={}
        for idx,s in enumerate(SS):
            if s in S_ids:
                S_ids[s].append(idx)
            else:
                S_ids[s]=[idx]
        
        R_ids={}
        for idx,r in enumerate(RS):
            if r in R_ids:
                R_ids[r].append(idx)
            else:
                R_ids[r]=[idx]
         
        W_ids={}
        for idx,w in enumerate(WS):
            if w in W_ids:
                W_ids[w].append(idx)
            else:
                W_ids[w]=[idx]
    
        return W_ids,D_ids,S_ids,R_ids
 
    def initialize_randint(self,K,D,V,L,N,gamma_default=1e-2,range=(1,10)):
        # Initialize parameters randomly
        # EM parameters
        omega=np.random.randint(range[0],range[1],(K,D))
        omega=omega/omega.sum(axis=1)[:,np.newaxis]
        phi=np.random.randint(range[0],range[1],(K,V))
        phi=phi/phi.sum(axis=1)[:,np.newaxis]
        pi=np.random.randint(range[0],range[1],K)
        pi=pi/pi.sum() 
        # VI parameters
        gamma=np.repeat(gamma_default,K**2)
        delta=np.random.randint(range[0],range[1],(L,K**2))
        delta=delta/delta.sum(axis=1)[:,np.newaxis]
        epsilon=np.random.randint(range[0],range[1],(N,K))
        epsilon=epsilon/epsilon.sum(axis=1)[:,np.newaxis]  
        
        return omega,phi,pi,gamma,delta,epsilon 
    
    
    def initialize_dir(self,beta,K,D,V,L,N):
        # Initialize EM parameters randomly (use diriclet to ensure normalization)
        omega=np.zeros((K,D))
        for k in range(K):
            omega[k,:]=stats.dirichlet.rvs(np.repeat(beta,D))
    
        phi=np.zeros((K,V))
        for k in range(K):
            phi[k,:]=stats.dirichlet.rvs(np.repeat(beta,V))
    
        pi=stats.dirichlet.rvs(np.repeat(beta,K)).flatten()
    
        # initialize VI parameters
        gamma=stats.dirichlet.rvs(np.repeat(beta,K**2))

        delta=np.zeros((L,K**2))
        for l in range(L):
            delta[l,:]=stats.dirichlet.rvs(np.repeat(beta,K**2))

        epsilon=np.zeros((N,K))
        for n in range(N):
            epsilon[n,:]=stats.dirichlet.rvs(np.repeat(beta,K))
    
        return omega,phi,pi,gamma,delta,epsilon 
    
    
    def start(self,K,initialize,alpha=1e-2,n_iter_EM=30,n_iter_VI=100,gamma_max_gap=1e-3,phi_max_gap=1e-3,gamma_default=1e-2,int_range=(1,10),beta=1e-1,verbose=10):
        # transfer
        WS,DS,SS,RS=self.transfer(self.G,self.texts)
        # initiate
        D=self.texts.shape[0]
        V=self.texts.shape[1]
        L=len(SS)
        N=len(WS)
        # helpers
        W_ids,D_ids,S_ids,R_ids=self.initialize_helpers(WS,DS,SS,RS)
        
        '''
        # Initialize parameters randomly
        # EM parameters
        omega=np.random.randint(1,10,(K,D))
        omega=omega/omega.sum(axis=1)[:,np.newaxis]
        phi=np.random.randint(1,10,(K,V))
        phi=phi/phi.sum(axis=1)[:,np.newaxis]
        pi=np.random.randint(1,10,K)
        pi=pi/pi.sum() 
        # VI parameters
        gamma=np.repeat(1e-2,K**2)
        delta=np.random.randint(1,10,(L,K**2))
        delta=delta/delta.sum(axis=1)[:,np.newaxis]
        epsilon=np.random.randint(1,10,(N,K))
        epsilon=epsilon/epsilon.sum(axis=1)[:,np.newaxis]
        '''
        
        # Initialize parameters randomly
        if initialize=='randint':
            omega,phi,pi,gamma,delta,epsilon=self.initialize_randint(K,D,V,L,N,gamma_default,int_range)
        elif initialize=='dir':
            omega,phi,pi,gamma,delta,epsilon=self.initialize_dir(beta,K,D,V,L,N)
        
        
        # variational-EM
        for it_em in range(n_iter_EM):
            # E-step
            for it_vi in range(n_iter_VI):
                gamma_last=gamma[:]
                # solve gamma&delta
                gamma=delta.sum(axis=0)+alpha

                for k in range(K**2):
                    delta[:,k]=omega[index2ij(k,K)[0],SS[range(L)]]*omega[index2ij(k,K)[1],RS[range(L)]]*np.exp(digamma(gamma[k]))
                delta=delta/delta.sum(axis=1)[:,np.newaxis]
                
                # check convergence 
                gamma_gap=np.abs(gamma-gamma_last).sum()
                if it_vi%verbose==0:
                    print('EM:%d,VI:%d,gamma_gap:%f'%(it_em,it_vi,gamma_gap))                
                if gamma_gap<gamma_max_gap:
                    break
        
            # solve epsilon  
            for k in range(K):
                epsilon[:,k]=omega[k,DS[range(N)]]*phi[k,WS[range(N)]]*pi[k]
            epsilon=epsilon/epsilon.sum(axis=1)[:,np.newaxis]

            # M-step
            phi_last=phi.copy()
            pi_last=pi[:]
            omega_last=omega.copy() 
            # omega
            S_dist=np.zeros((L,K))
            R_dist=np.zeros((L,K))
            for l in range(L):
                S_dist[l,:]=delta[l,:].reshape(K,K).sum(axis=1)
                R_dist[l,:]=delta[l,:].reshape(K,K).sum(axis=0)
            term_1=term_2=term_3=0
    
            for d in range(D):
                ep_id=D_ids.get(d,[])
                term_1=epsilon[ep_id,:].sum(axis=0)
                S_id=S_ids.get(d,[])
                term_2=S_dist[S_id,:].sum(axis=0)
                R_id=R_ids.get(d,[])
                term_3=R_dist[R_id,:].sum(axis=0)
                omega[:,d]=term_1+term_2+term_3
            omega=omega/omega.sum(axis=1)[:,np.newaxis]
            
            # phi
            for w in range(V):
                ep_id=W_ids.get(w,[])
                phi[:,w]=epsilon[ep_id,:].sum(axis=0)
            phi=phi/phi.sum(axis=1)[:,np.newaxis]

            # pi
            pi=epsilon.sum(axis=0)
            pi=pi/pi.sum()
            
            # check convergence
            phi_gap=np.abs(phi-phi_last).sum()
            pi_gap=np.abs(pi-pi_last).sum()
            omega_gap=np.abs(omega-omega_last).sum() 
            print('EM:%d,phi_gap:%f,pi_gap:%f,omega_gap:%f'%(it_em,phi_gap,pi_gap,omega_gap))
            if phi_gap<phi_max_gap:
                break 
            
            
        # one more VI after EM converges
        for it_vi in range(n_iter_VI):
            gamma_last=gamma[:]
            # solve gamma&delta
            gamma=delta.sum(axis=0)+alpha

            for k in range(K**2):
                delta[:,k]=omega[index2ij(k,K)[0],SS[range(L)]]*omega[index2ij(k,K)[1],RS[range(L)]]*np.exp(digamma(gamma[k]))
            delta=delta/delta.sum(axis=1)[:,np.newaxis]
                
            # check convergence 
            gamma_gap=np.abs(gamma-gamma_last).sum()
            if it_vi%verbose==0:
                print('FINAL:VI:%d,gamma_gap:%f'%(it_vi,gamma_gap))                

        
        # derive expectation from posterior (variational) distribution
        theta=gamma/gamma.sum()
        theta=theta.reshape(K,K)
        
        return theta,omega,phi,pi

