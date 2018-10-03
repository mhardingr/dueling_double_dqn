# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import scipy.stats as ss 
 
class custom_distribution(ss.rv_continuous):
     "Custom distribution"
     def _pdf(self, x):
         return 0.5*(1+x)

class custom_distribution_1(ss.rv_continuous):
     "Custom distribution"
     def _pdf(self, x):
         return 15/16*(x**2)*(1+x)**2




    
def Q4_2(sample_sizes, custom_distribution,seed):
    rv= custom_distribution(a=-1, b=1, name='custom_distribution')
    print("############################################################")
    print("Q4.2")
    for n in sample_sizes:
        np.random.seed(seed)
        sample=rv.rvs(size=n)
        f=[1.5*x*x*(1+x) for x in sample]
        m= sum(f)/len(f)
        v=np.var(f)
        v_= sum((x - m) ** 2 for x in f) / len(f)
        print("sample mean and variance of size ",n," =",m,v)
    
#print(v_10000_)

def Q4_3(sample_sizes,custom_distribution,custom_distribution_1,seed):
    rv= custom_distribution(a=-1, b=1, name='custom_distribution')
    print("############################################################")
    print("Q4.3")
    print("q(x)=N(3,1)")
    for n in sample_sizes:
        np.random.seed(seed)
        rv_N31=ss.norm(3,1)
        sample=np.random.normal(3,1,size=n)
        s=[1.5*x*x*(1+x) for x in sample]
        weights=rv.pdf(sample)/rv_N31.pdf(sample)
        EF=np.dot(s,weights)/len(s)
        v=np.var(np.multiply(s,weights))
        print("sample mean and variance of size", n ,"by importance sampling with q(x)=N(3,1)=",EF,v)
    print("#######################")      
    print("q(x)=N(0,1)")
    for n in sample_sizes:
        np.random.seed(seed)
        rv_N01=ss.norm(0,1)
        sample=np.random.normal(0,1,size=n)
        s=[1.5*x*x*(1+x) for x in sample]
        weights=rv.pdf(sample)/rv_N01.pdf(sample)
        EF=np.dot(s,weights)/len(s)
        v=np.var(np.multiply(s,weights))
        print("sample mean and variance of size", n ,"by importance sampling with q(x)=N(0,1)=",EF,v)
    print("#######################")
    print("q(x)=15/16*(x**2)*(1+x)**2")
    for n in sample_sizes:
        np.random.seed(seed)
        rv3= custom_distribution_1(a=-1, b=1, name='custom_distribution_1')
        np.random.seed(seed)
        sample=rv3.rvs(size=n)
        s=[1.5*x*x*(1+x) for x in sample]
        weights=rv.pdf(sample)/rv3.pdf(sample)
        EF=np.dot(s,weights)/len(s)
        v=np.var(np.multiply(s,weights))
        print("sample mean and variance of size", n ,"by importance sampling with q(x)=q(x)=15/16*(x**2)*(1+x)**2=",EF,v)
    

def Q4_4(sample_sizes,custom_distribution,custom_distribution_1,seed):
    rv= custom_distribution(a=-1, b=1, name='custom_distribution')
    print("############################################################")
    print("Q4.4")
    print("q(x)=N(3,1)")
    for n in sample_sizes:
        np.random.seed(seed)
        rv_N31=ss.norm(3,1)
        sample=np.random.normal(3,1,size=n)
        s=[1.5*x*x*(1+x) for x in sample]
        weights=rv.pdf(sample)/rv_N31.pdf(sample)
        normalized_weights = np.multiply(weights, np.mean(weights))
        EF=np.dot(s,normalized_weights)/len(s)
        v=np.var(np.multiply(s,normalized_weights))
        print("sample mean and variance of size", n ,"by importance sampling with q(x)=N(3,1)=",EF,v)
    print("#######################")      
    print("q(x)=N(0,1)")
    for n in sample_sizes:
        np.random.seed(seed)
        rv_N01=ss.norm(0,1)
        sample=np.random.normal(0,1,size=n)
        s=[1.5*x*x*(1+x) for x in sample]
        weights=rv.pdf(sample)/rv_N01.pdf(sample)
        normalized_weights = np.multiply(weights, np.mean(weights))
        EF=np.dot(s,normalized_weights)/len(s)
        v=np.var(np.multiply(s,normalized_weights))
        print("sample mean and variance of size", n ,"by importance sampling with q(x)=N(0,1)=",EF,v)
    print("#######################")
    print("q(x)=15/16*(x**2)*(1+x)**2")
    for n in sample_sizes:
        np.random.seed(seed)
        rv3= custom_distribution_1(a=-1, b=1, name='custom_distribution_1')
        np.random.seed(seed)
        sample=rv3.rvs(size=n)
        s=[1.5*x*x*(1+x) for x in sample]
        weights=rv.pdf(sample)/rv3.pdf(sample)
        normalized_weights = np.multiply(weights, np.mean(weights))
        EF=np.dot(s,normalized_weights)/len(s)
        v=np.var(np.multiply(s,normalized_weights))
        print("sample mean and variance of size", n ,"by importance sampling with q(x)=q(x)=15/16*(x**2)*(1+x)**2=",EF,v)    

seed=10703
Q4_2([10, 1000, 10000], custom_distribution,seed)
Q4_3([10, 1000, 10000],custom_distribution,custom_distribution_1,seed)
Q4_4([10, 1000, 10000],custom_distribution,custom_distribution_1,seed)