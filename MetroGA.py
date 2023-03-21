
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
import os
import sys
import csv


#ststs
#Lo spazio è lineare!(Sums and difference of vectors and multiplication of a vector by a real number are defined) perchè le euqazioni sono tutte definite nello spazio euclideo!!!!!!
#the general problem of calculate the posterion density probability is: pag 32
#the general problem of calculate the posterion density probability for a inverse problem is: pag 34, two cases for the likelihood: probabilistic forward model (34) and functional forward model (35)
#Least-squares techniques arise when all the ‘input’ probability densities are assumed to be Gaussian (the formulation with exp and misfit(pag 64) in this case the maximum Likehood point equal to #maximize the sigma. 

# Define the N-dimensional probability distribution function
def Sigmap(x,covd,covm,mud,mum,t,hoP,hoC,hoSc, yP, yC, ySc):
    
    covD= covd + 0
    gmox = [ ]
    gmshc = [ ]
    gmshSc = [ ]
    gm = [ ]
    hP = hoP 
    hC = hoC
    hSc = hoSc
          
    for i in range (len(t)):
        if t[i] == 0 or t[i] == 1:
           gmox.append(abs((1 - np.exp(-x[4] * t[i])) * ((1-x[5]) * (hoP - yP[i]))))
           gmshc.append(abs((1 - np.exp(-x[0] * t[i])) * ((1-x[1]) * (hoC - yC[i])))) 
           gmshSc.append(((1 - np.exp(-x[2] * t[i])) * ((1-x[3]) * (hoSc - ySc[i]))))
           gm.append(gmox[i] + gmshc[i] + gmshSc[i])

        else:
            hP -= gmox[i-1]
            hC -= gmshc[i-1]
            hSc -= gmshSc[i-1]                                       
            gmox.append(abs((1 - np.exp(-x[4] * t[i])) * (hP - x[5] * hoP + yP[i] * (x[5] -1))))   
            gmshc.append(abs((1 - np.exp(-x[0] * t[i])) * (hC - x[1] * hoC + yC[i] * (x[1] -1))))
            gmshSc.append(((1 - np.exp(-x[2] * t[i])) * (hSc -x[3] * hoSc + ySc[i] * (x[3] -1))))     
            gm.append(gmox[i] + gmshc[i] + gmshSc[i])
         
    msfit = 0.5 * (np.matmul((np.matmul((gm - mud),np.linalg.inv(covD))), (gm - mud).T) + np.matmul((np.matmul((x - mum).T, np.linalg.inv(covm))), (x - mum)))
    return np.exp(-msfit)
    
    

def metropolis3(N3, num_steps3, delta3,mum,covm,mud,covd,x,t,hoP,hoC,hoSc, yP, yC, ySc):
    samples3 = [ ]        
    for i in range(num_steps3):
        # Propose a new point
        x_new = x + delta3*np.random.randn(N3)        
        #PDE Alternatives for sampling
        #np.random.random_sample()        
        #np.random.uniform(-1,1)        
        if  Sigmap(x_new,covd,covm,mud,mum,t,hoP,hoC,hoSc,yP, yC, ySc) >= Sigmap(x,covd,covm,mud,mum,t,hoP,hoC,hoSc, yP, yC, ySc):
            samples3.append(x_new)
            x = x_new            
        else:           
            if np.random.rand() < Sigmap(x_new,covd,covm,mud,mum,t,hoP,hoC,hoSc, yP, yC, ySc)/Sigmap(x,covd,covm,mud,mum,t,hoP,hoC,hoSc, yP, yC, ySc):
                samples3.append(x_new)
                x = x_new
    Prob3 = np.zeros(len(samples3))    
    #Calculate the probability of each samples extracted       
    for i in range (len(samples3)):
        Prob3[i] = Sigmap(samples3[i],covd,covm,mud,mum,t,hoP,hoC,hoSc, yP, yC, ySc)
           
    acceptance_rate3 = len(samples3) / num_steps3
    print("Acceptance rate is:")
    print(acceptance_rate3)
    print("The mumber of the samples are")
    print(len(samples3))

    return samples3,Prob3



def ForwardModel(samples3,devstd,t,hoP,hoC,hoSc,yP, yC, ySc,mud):
                    
    gmox = np.zeros((len(samples3),len(t)))
    gmshc = np.zeros((len(samples3),len(t)))
    gmshSc = np.zeros((len(samples3),len(t)))
    gm = np.zeros((len(samples3),len(t)))
    hP = hoP 
    hC = hoC
    hSc = hoSc
       
    for i in range(len(samples3)):
        for j in range (len(t)):
            if t[j] == 0 or t[j] == 1:
               gmox[i][j] = (abs((1 - np.exp(-samples3[i][4] * t[j])) * ((1-samples3[i][5]) * (hoP - yP[j]))))
               gmshc[i][j] = (abs((1 - np.exp(-samples3[i][0] * t[j])) * ((1-samples3[i][1]) * (hoC - yC[j])))) 
               gmshSc[i][j] = (abs((1 - np.exp(-samples3[i][2] * t[j])) * ((1-samples3[i][3]) * (hoSc - ySc[j]))))
               gm[i][j] = (gmox[i][j] + gmshc[i][j] + gmshSc[i][j])
            else:
                hP -= gmox[i][j-1]
                hC -= gmshc[i][j-1]
                hSc -= gmshSc[i][j-1]                            
                gmox[i][j]= (abs((1 - np.exp(-samples3[i][4] * t[j])) * (hP - samples3[i][5] * hoP + yP[j] * (samples3[i][5] -1))))   
                gmshc[i][j] = (abs((1 - np.exp(-samples3[i][0] * t[j])) * (hC - samples3[i][1] * hoC + yC[j] * (samples3[i][1] -1))))
                gmshSc[i][j]= (abs((1 - np.exp(-samples3[i][2] * t[j])) * (hSc -samples3[i][3] * hoSc + ySc[j] * (samples3[i][3] -1))))     
                gm[i][j]= (gmox[i][j] + gmshc[i][j] + gmshSc[i][j])
        hP = hoP 
        hC = hoC
        hSc = hoSc

    return gm



def FindNewStartingPoint(Prob3,samples3):

    MaxProb3 = max(Prob3)
    for i in range(len(Prob3)):
        if Prob3[i] == MaxProb3:
           IndexMaxProb = i
           break       
    print("Nuovo punto di partenza:",samples3[IndexMaxProb])

    return samples3[IndexMaxProb]



def SaveOutput(samples3, gm2, Prob3):

    # Define the filename for the CSV file 
    filenameProb3 = "Prob3.csv"
    filenameGm2 = "Forward.csv"
    filenameSamples = "Samples.csv"
   
    # Write the Forward to the CSV file
    with open(filenameGm2, 'w', newline='') as csvfile:
         csvwriter = csv.writer(csvfile)
         for row in gm2:
             csvwriter.writerow(row)

    # Write the samples to the CSV file
    with open(filenameSamples, 'w', newline='') as csvfile:
         csvwriter = csv.writer(csvfile)
         for row in samples3:
             csvwriter.writerow(row)

    # Write the Prob to the CSV file
    with open(filenameProb3, 'w', newline='') as csvfile:
         csvwriter = csv.writer(csvfile)
         csvwriter.writerow(Prob3)
         
         
def Plot(gm,t,yP,yC,ySc,mud,devstd,samples3,Prob3):
                
    plt.figure(1)
    plt.title("Forward model posterior vs data")
    plt.plot(t,-gm.T) 
    plt.errorbar(t, -mud, yerr=devstd,label='dati')
    plt.legend()
    
    plt.figure(2)
    plt.title("Groundwater level for each litology")
    plt.plot(t, yP, label='Groundwater level for Peat')   
    plt.plot(t,yC,label='Groundwater level for Clay')  
    plt.plot(t,ySc,label='Groundwater level for Sandy clay')
    plt.legend()
    
    fig, bx = plt.subplots()
               
    plt.figure(3)
    bx.set_title('Metropolis Algorithm Samples for data')
    bx.scatter([s[0] for s in samples3], [s[1] for s in samples3], s=1)
    bx.set_xlabel('x')
    bx.set_ylabel('y')
    
    plt.figure(4)
    plt.title("Prability of the functions on each samples for data")
    plt.plot([i for i in range(len(samples3))], Prob3)

    plt.show()  

#=======================================================================================
#=======================================================================================
#=======================================================================================
#=================Main==================================================================
#=======================================================================================
#=======================================================================================

def main():
    

#==========================================================
#==Sampling Posterior Density probability(metropolis3)===== 
#==========================================================

#Further investigations:
# diagonal covariance matrice neglet informations and indipedence of sampling

#Time
    t = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

#Parameter of Forward model
    hoP = 0.7
    hoC = 0.8
    hoSc = 3

#Parameters of metropolis
    N3 = 6
    num_steps3 = 666
    delta3 = 0.0011
    x = np.array([0.01739949, 0.60231223, 0.00259897, 0.6211021,  0.00836138, 0.91836421])

#[0.0188013,  0.5958105,  0.00245935, 0.59172465, 0.01059123, 0.90085271]) linear water 0.0011
#[0.02,0.6,0.02,0.6,0.01,0.9]) #Starting point)
#[0.01739949, 0.60231223, 0.00259897, 0.6211021,  0.00836138, 0.91836421]) #sin water 0.0011

#Prior Model

#Vsh CLAy,Rh clay, vsh snady clay, rh sandy clay, vox peat, rh peat
    mum = np.array([0.02,0.6,0.02,0.6,0.01,0.9])
    devstm= np.array([0.005,0.05,0.005,0.05,0.005,0.05])
    #Sampling a normal distribution for calculate the covariance mantrix on prior parameteres
    Ncm = 200 # number of samples to generate
#generate N samples from the normal distribution
    samplesm= np.zeros((Ncm,len(mum)))
    for i in range(len(mum)):
        samplesm[:,i] = np.random.normal(mum[i], devstm[i], Ncm)
    covm = np.cov(samplesm.T)


#Data
    s1 = 0.01 
    s2 = 0.005 
    s3 = 0.01 
    s4 = 0.01 
    s5 = 0.01 
    s6 = 0.005 
    s7 = 0.01 
    s8 = 0.01 
    s9 = 0.01 
    s10 = 0.005
    s11 = 0.01 
    s12 = 0.01 
    s13 = 0.005 
    s14 = 0.01 
    s15 = 0.005 
    s16 = 0.01 
    s17 = 0.01 
    s18 = 0.005 
    s19 = 0.01 
    s20 = 0.005
    s21 = 0.01 
    mud = np.array([0,0.003,0.001,0.006,0.004,0.008,0.007,0.01,0.008,0.014,0.009,0.018,0.012,0.020,0.015,0.023,0.017,0.025,0.018,0.028,0.020])  
    devstd=np.array([s1,s1,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21])  
 #Sampling a normal distribution for calculate the covariance mantrix on prior parameteres
    Ncd = 200 # number of samples to generate
# generate N samples from the normal distribution
    samplesd= np.zeros((Ncd,len(mud)))
    for i in range(len(mud)):    
        samplesd[:,i] = np.random.normal(mud[i], devstd[i], Ncd)
    covd = np.cov(samplesd.T)


 
#========= Groundwater level====================

#Fixed value
#hwP = 0.7
#hwC = 0.8
#hwSc = 3     
   
#Linear trend for hwet:
#h wet Sandy clay
    hwSc=np.array([2.5,1])
    tfitSc=np.array([0,20])
    B = np.stack([tfitSc, np.ones(len(tfitSc))]).T
    mSc, cSc = np.linalg.lstsq(B, hwSc, rcond=None)[0]

#h wet Peat
    hwP=np.array([0.7,0.7])
    tfitP=np.array([0,20]) 
    C = np.stack([tfitP, np.ones(len(tfitP))]).T
    mP, cP = np.linalg.lstsq(C, hwP, rcond=None)[0]

#h wet clay
    hwC=np.array([0.8,0.8])
    tfitC=np.array([0,20])
    A = np.stack([tfitC, np.ones(len(tfitC))]).T
    mC, cC = np.linalg.lstsq(A, hwC, rcond=None)[0]

    yC = mC*t + cC
#ySc = mSc*t + cSc
    yP = mP*t + cP

#Sin trend for hwet:
#h wet Sandy clay
    ySc=mSc * t +cSc + 0.2 * np.sin(t-np.pi)


#Calling metropolis3
    samples3,Prob3 = metropolis3(N3, num_steps3, delta3, mum, covm, mud, covd, x,t, hoP, hoC, hoSc, yP, yC, ySc)



#==========================================================
#==============Only Forward model========================== 
#==========================================================
    gm = ForwardModel(samples3,devstd,t,hoP,hoC,hoSc,yP, yC, ySc,mud)


#====================================================
#===========NewStartingPoint=========================
#====================================================
    NewStartingPoint = FindNewStartingPoint(Prob3,samples3)

#====================================================
#============Save in csv=============================
#====================================================
    SaveOutput(samples3, gm, Prob3)

#====================================================
#==================Plot==============================
#====================================================
    Plot(gm,t,yP,yC,ySc,mud,devstd,samples3,Prob3)



if __name__ == "__main__":
    main()




