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
rv= custom_distribution(a=-1, b=1, name='custom_distribution')

class custom_distribution_1(ss.rv_continuous):
     "Custom distribution"
     def _pdf(self, x):
         return 15/16*(x**2)*(1+x)**2
def Q4_2():
    print("############################################################")
    print("Q4.2")
    sample_10=rv.rvs(size=10)
    s_10=[1.5*x*x*(1+x) for x in sample_10]
    m_10= sum(s_10)/len(s_10)
    v_10=np.var(s_10)
    v_10_= sum((x - m_10) ** 2 for x in s_10) / len(s_10)
    print("sample mean of size 10 =",m_10)
    print("sample variance of size 10 =",v_10)
    #print(v_10_)
    print("#######################")
    sample_1000=rv.rvs(size=1000)
    s_1000=[1.5*x*x*(1+x) for x in sample_1000]
    m_1000=sum(s_1000)/len(s_1000)
    v_1000=np.var(s_1000)
    v_1000_= sum((x - m_1000) ** 2 for x in s_1000) / len(s_1000)
    print("sample mean of size 1000 =",m_1000)
    print("sample variance of size 1000 =",v_1000)
    #print(v_1000_)
    print("#######################")
    sample_10000=rv.rvs(size=10000)
    s_10000=[1.5*x*x*(1+x) for x in sample_10000]
    m_10000=sum(s_10000)/len(s_10000)
    v_10000=np.var(s_10000)
    v_10000_= sum((x - m_10000) ** 2 for x in s_10000) / len(s_10000)
    print("sample mean of size 10000 =",m_10000)
    print("sample variance of size 10000 =",v_10000)
#print(v_10000_)

def Q4_3():
    print("############################################################")
    print("Q4.3")
    print("q(x)=N(3,1)")
    rv_N31=ss.norm(3,1)
    sample_N31_10=np.random.normal(3,1,size=10)
    s_N31_10=[1.5*x*x*(1+x) for x in sample_N31_10]
    weights_N31_10=rv.pdf(sample_N31_10)/rv_N31.pdf(sample_N31_10)
    EF_N31_10=np.dot(s_N31_10,weights_N31_10)/len(s_N31_10)
    print("sample mean of size 10 by importance sampling with q(x)=N(3,1)=",EF_N31_10)
    v_N31_10=np.var(np.multiply(s_N31_10,weights_N31_10))
    print("sample variance of size 10 by importance sampling with q(x)=N(3,1)=",v_N31_10)
    
    sample_N31_1000=np.random.normal(3,1,size=1000)
    s_N31_1000=[1.5*x*x*(1+x) for x in sample_N31_1000]
    weights_N31_1000=rv.pdf(sample_N31_1000)/rv_N31.pdf(sample_N31_1000)
    EF_N31_1000=np.dot(s_N31_1000,weights_N31_1000)/len(s_N31_1000)
    print("sample mean of size 1000 by importance sampling with q(x)=N(3,1)=",EF_N31_1000)
    v_N31_1000=np.var(np.multiply(s_N31_1000,weights_N31_1000))
    print("sample variance of size 1000 by importance sampling with q(x)=N(3,1)=",v_N31_1000)
    
    sample_N31_10000=np.random.normal(3,1,size=10)
    s_N31_10000=[1.5*x*x*(1+x) for x in sample_N31_10000]
    weights_N31_10000=rv.pdf(sample_N31_10000)/rv_N31.pdf(sample_N31_10000)
    EF_N31_10000=np.dot(s_N31_10000,weights_N31_10000)/len(s_N31_10000)
    print("sample mean of size 10000 by importance sampling with q(x)=N(3,1)=",EF_N31_10000)
    v_N31_10000=np.var(np.multiply(s_N31_10000,weights_N31_10000))
    print("sample variance of size 10000 by importance sampling with q(x)=N(3,1)=",v_N31_10000)
    print("#######################")
    print("q(x)=N(0,1)")
    rv_N01=ss.norm(0,1)
    sample_N01_10=np.random.normal(0,1,size=10)
    s_N01_10=[1.5*x*x*(1+x) for x in sample_N01_10]
    weights_N01_10=rv.pdf(sample_N01_10)/rv_N01.pdf(sample_N01_10)
    EF_N01_10=np.dot(s_N01_10,weights_N01_10)/len(s_N01_10)
    print("sample mean of size 10 by importance sampling with q(x)=N(0,1)=",EF_N01_10)
    v_N01_10=np.var(np.multiply(s_N01_10,weights_N01_10))
    print("sample variance of size 10 by importance sampling with q(x)=N(0,1)=",v_N01_10)
    
    sample_N01_1000=np.random.normal(0,1,size=1000)
    s_N01_1000=[1.5*x*x*(1+x) for x in sample_N01_1000]
    weights_N01_1000=rv.pdf(sample_N01_1000)/rv_N01.pdf(sample_N01_1000)
    EF_N01_1000=np.dot(s_N01_1000,weights_N01_1000)/len(s_N01_1000)
    print("sample mean of size 1000 by importance sampling with q(x)=N(0,1)=",EF_N01_1000)
    v_N01_1000=np.var(np.multiply(s_N01_1000,weights_N01_1000))
    print("sample variance of size 1000 by importance sampling with q(x)=N(0,1)=",v_N01_1000)
    
    sample_N01_10000=np.random.normal(0,1,size=10000)
    s_N01_10000=[1.5*x*x*(1+x) for x in sample_N01_10000]
    weights_N01_10000=rv.pdf(sample_N01_10000)/rv_N01.pdf(sample_N01_10000)
    EF_N01_10000=np.dot(s_N01_10000,weights_N01_10000)/len(s_N01_10000)
    print("sample mean of size 10000 by importance sampling with q(x)=N(0,1)=",EF_N01_10000)
    v_N01_10000=np.var(np.multiply(s_N01_10000,weights_N01_10000))
    print("sample variance of size 10000 by importance sampling with q(x)=N(0,1)=",v_N01_10000)
    
    print("#######################")
    print("q(x)=15/16*(x**2)*(1+x)**2")
    rv3= custom_distribution_1(a=-1, b=1, name='custom_distribution_1')
    sample_cd1_10=rv3.rvs(size=10)
    s_cd1_10=[1.5*x*x*(1+x) for x in sample_cd1_10]
    weights_cd1_10=rv.pdf(sample_cd1_10)/rv3.pdf(sample_cd1_10)
    EF_cd1_10=np.dot(s_cd1_10,weights_cd1_10)/len(s_cd1_10)
    print("sample mean of size 10 by importance sampling with q(x)=q(x)=15/16*(x**2)*(1+x)**2=",EF_cd1_10)
    v_cd1_10=np.var(np.multiply(s_cd1_10,weights_cd1_10))
    print("sample variance of size 10 by importance sampling with q(x)=15/16*(x**2)*(1+x)**2=",v_cd1_10)
    
    sample_cd1_1000=rv3.rvs(size=1000)
    s_cd1_1000=[1.5*x*x*(1+x) for x in sample_cd1_1000]
    weights_cd1_1000=rv.pdf(sample_cd1_1000)/rv3.pdf(sample_cd1_1000)
    EF_cd1_1000=np.dot(s_cd1_1000,weights_cd1_1000)/len(s_cd1_1000)
    print("sample mean of size 1000 by importance sampling with q(x)=q(x)=15/16*(x**2)*(1+x)**2=",EF_cd1_1000)
    v_cd1_1000=np.var(np.multiply(s_cd1_1000,weights_cd1_1000))
    print("sample variance of size 1000 by importance sampling with q(x)=15/16*(x**2)*(1+x)**2=",v_cd1_1000)
    
    sample_cd1_10000=rv3.rvs(size=10000)
    s_cd1_10000=[1.5*x*x*(1+x) for x in sample_cd1_10000]
    weights_cd1_10000=rv.pdf(sample_cd1_10000)/rv3.pdf(sample_cd1_10000)
    EF_cd1_10000=np.dot(s_cd1_10000,weights_cd1_10000)/len(s_cd1_10000)
    print("sample mean of size 10000 by importance sampling with q(x)=q(x)=15/16*(x**2)*(1+x)**2=",EF_cd1_10000)
    v_cd1_10000=np.var(np.multiply(s_cd1_10000,weights_cd1_10000))
    print("sample variance of size 10 by importance sampling with q(x)=15/16*(x**2)*(1+x)**2=",v_cd1_10000)

def Q4_4():
    print("############################################################")
    print("Q4.4")
    print("q(x)=N(3,1)")
    rv_N31=ss.norm(3,1)
    sample_N31_10=np.random.normal(3,1,size=10)
    s_N31_10=[1.5*x*x*(1+x) for x in sample_N31_10]
    weights_N31_10=rv.pdf(sample_N31_10)/rv_N31.pdf(sample_N31_10)
    normalized_weights_N31_10 = np.multiply(weights_N31_10, np.mean(weights_N31_10))
    EF_N31_10=np.dot(s_N31_10,normalized_weights_N31_10)/len(s_N31_10)
    print("sample mean of size 10 by weighted importance sampling with q(x)=N(3,1)=",EF_N31_10)
    v_N31_10=np.var(np.multiply(s_N31_10,normalized_weights_N31_10))
    print("sample variance of size 10 by weighted importance sampling with q(x)=N(3,1)=",v_N31_10)
    
    sample_N31_1000=np.random.normal(3,1,size=1000)
    s_N31_1000=[1.5*x*x*(1+x) for x in sample_N31_1000]
    weights_N31_1000=rv.pdf(sample_N31_1000)/rv_N31.pdf(sample_N31_1000)
    normalized_weights_N31_1000 = np.multiply(weights_N31_1000, np.mean(weights_N31_1000))
    EF_N31_1000=np.dot(s_N31_1000,normalized_weights_N31_1000)/len(s_N31_1000)
    print("sample mean of size 1000 by weighted importance sampling with q(x)=N(3,1)=",EF_N31_1000)
    v_N31_1000=np.var(np.multiply(s_N31_1000,normalized_weights_N31_1000))
    print("sample variance of size 1000 by weighted importance sampling with q(x)=N(3,1)=",v_N31_1000)
    
    sample_N31_10000=np.random.normal(3,1,size=10)
    s_N31_10000=[1.5*x*x*(1+x) for x in sample_N31_10000]
    weights_N31_10000=rv.pdf(sample_N31_10000)/rv_N31.pdf(sample_N31_10000)
    normalized_weights_N31_10000 = np.multiply(weights_N31_10000, np.mean(weights_N31_10000))
    EF_N31_10000=np.dot(s_N31_10000,normalized_weights_N31_10000)/len(s_N31_10000)
    print("sample mean of size 10000 by weighted importance sampling with q(x)=N(3,1)=",EF_N31_10000)
    v_N31_10000=np.var(np.multiply(s_N31_10000,normalized_weights_N31_10000))
    print("sample variance of size 10000 by weighted importance sampling with q(x)=N(3,1)=",v_N31_10000)
    print("#######################")
    print("q(x)=N(0,1)")
    rv_N01=ss.norm(0,1)
    sample_N01_10=np.random.normal(0,1,size=10)
    s_N01_10=[1.5*x*x*(1+x) for x in sample_N01_10]
    weights_N01_10=rv.pdf(sample_N01_10)/rv_N01.pdf(sample_N01_10)
    normalized_weights_N01_10 = np.multiply(weights_N01_10, np.mean(weights_N01_10))
    EF_N01_10=np.dot(s_N01_10,normalized_weights_N01_10)/len(s_N01_10)
    print("sample mean of size 10 by weighted importance sampling with q(x)=N(0,1)=",EF_N01_10)
    v_N01_10=np.var(np.multiply(s_N01_10,normalized_weights_N01_10))
    print("sample variance of size 10 by weighted importance sampling with q(x)=N(0,1)=",v_N01_10)
    
    sample_N01_1000=np.random.normal(0,1,size=1000)
    s_N01_1000=[1.5*x*x*(1+x) for x in sample_N01_1000]
    weights_N01_1000=rv.pdf(sample_N01_1000)/rv_N01.pdf(sample_N01_1000)
    normalized_weights_N01_1000 = np.multiply(weights_N01_1000, np.mean(weights_N01_1000))
    EF_N01_1000=np.dot(s_N01_1000,normalized_weights_N01_1000)/len(s_N01_1000)
    print("sample mean of size 1000 by weighted importance sampling with q(x)=N(0,1)=",EF_N01_1000)
    v_N01_1000=np.var(np.multiply(s_N01_1000,normalized_weights_N01_1000))
    print("sample variance of size 1000 by weighted importance sampling with q(x)=N(0,1)=",v_N01_1000)
    
    sample_N01_10000=np.random.normal(0,1,size=10000)
    s_N01_10000=[1.5*x*x*(1+x) for x in sample_N01_10000]
    weights_N01_10000=rv.pdf(sample_N01_10000)/rv_N01.pdf(sample_N01_10000)
    normalized_weights_N01_10000 = np.multiply(weights_N01_10000, np.mean(weights_N01_10000))
    EF_N01_10000=np.dot(s_N01_10000,normalized_weights_N01_10000)/len(s_N01_10000)
    print("sample mean of size 10000 by weighted importance sampling with q(x)=N(0,1)=",EF_N01_10000)
    v_N01_10000=np.var(np.multiply(s_N01_10000,normalized_weights_N01_10000))
    print("sample variance of size 10000 by weighted importance sampling with q(x)=N(0,1)=",v_N01_10000)
    
    print("#######################")
    print("q(x)=15/16*(x**2)*(1+x)**2")
    rv3= custom_distribution_1(a=-1, b=1, name='custom_distribution_1')
    sample_cd1_10=rv3.rvs(size=10)
    s_cd1_10=[1.5*x*x*(1+x) for x in sample_cd1_10]
    weights_cd1_10=rv.pdf(sample_cd1_10)/rv3.pdf(sample_cd1_10)
    normalized_weights_N01_10 = np.multiply(weights_N01_10, np.mean(weights_N01_10))
    EF_cd1_10=np.dot(s_cd1_10,normalized_weights_N01_10)/len(s_cd1_10)
    print("sample mean of size 10 by weighted importance sampling with q(x)=q(x)=15/16*(x**2)*(1+x)**2=",EF_cd1_10)
    v_cd1_10=np.var(np.multiply(s_cd1_10,normalized_weights_N01_10))
    print("sample variance of size 10 by weighted importance sampling with q(x)=15/16*(x**2)*(1+x)**2=",v_cd1_10)
    
    sample_cd1_1000=rv3.rvs(size=1000)
    s_cd1_1000=[1.5*x*x*(1+x) for x in sample_cd1_1000]
    weights_cd1_1000=rv.pdf(sample_cd1_1000)/rv3.pdf(sample_cd1_1000)
    normalized_weights_N01_1000 = np.multiply(weights_N01_1000, np.mean(weights_N01_1000))
    EF_cd1_1000=np.dot(s_cd1_1000,normalized_weights_N01_1000)/len(s_cd1_1000)
    print("sample mean of size 1000 by weighted importance sampling with q(x)=q(x)=15/16*(x**2)*(1+x)**2=",EF_cd1_1000)
    v_cd1_1000=np.var(np.multiply(s_cd1_1000,normalized_weights_N01_1000))
    print("sample variance of size 1000 by weighted importance sampling with q(x)=15/16*(x**2)*(1+x)**2=",v_cd1_1000)
    
    sample_cd1_10000=rv3.rvs(size=10000)
    s_cd1_10000=[1.5*x*x*(1+x) for x in sample_cd1_10000]
    weights_cd1_10000=rv.pdf(sample_cd1_10000)/rv3.pdf(sample_cd1_10000)
    normalized_weights_N01_10000 = np.multiply(weights_N01_10000, np.mean(weights_N01_10000))
    EF_cd1_10000=np.dot(s_cd1_10000,normalized_weights_N01_10000)/len(s_cd1_10000)
    print("sample mean of size 10000 by weighted importance sampling with q(x)=q(x)=15/16*(x**2)*(1+x)**2=",EF_cd1_10000)
    v_cd1_10000=np.var(np.multiply(s_cd1_10000,normalized_weights_N01_10000))
    print("sample variance of size 10000 by weighted importance sampling with q(x)=15/16*(x**2)*(1+x)**2=",v_cd1_10000)

np.random.seed(0)
Q4_2()
np.random.seed(0)
Q4_3()
np.random.seed(0)
Q4_4()