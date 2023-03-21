
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
import csv
import json


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
    
    

def metropolis(N3, num_steps3, delta3,mum,covm,mud,covd,x,t,hoP,hoC,hoSc, yP, yC, ySc):
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
    print("Acceptance rate is: ", acceptance_rate3) 
    print("The mumber of the samples are ", len(samples3))
  
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
    print("Nuovo punto di partenza:", samples3[IndexMaxProb])

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

def CovarianceMatrix(mum,devstm,Ncm,Ncd,mud,devstd): 
    samplesm= np.zeros((Ncm,len(mum)))
    for i in range(len(mum)):
        samplesm[:,i] = np.random.normal(mum[i], devstm[i], Ncm)
    covm = np.cov(samplesm.T)

    samplesd= np.zeros((Ncd,len(mud)))
    for i in range(len(mud)):    
        samplesd[:,i] = np.random.normal(mud[i], devstd[i], Ncd)
    covd = np.cov(samplesd.T)
    
    return covd,covm


def Groundwaterlevel(t,hwSc,hwP,hwC,Choose,hoP,hoSc,hoC):
    
    tfit =np.array([t[0],t[(len(t)-1)]])

    #Parameters of straight line
    B = np.stack([tfit, np.ones(len(tfit))]).T
    mSc, cSc = np.linalg.lstsq(B, hwSc, rcond=None)[0]
    C = np.stack([tfit, np.ones(len(tfit))]).T
    mP, cP = np.linalg.lstsq(C, hwP, rcond=None)[0]   
    A = np.stack([tfit, np.ones(len(tfit))]).T
    mC, cC = np.linalg.lstsq(A, hwC, rcond=None)[0]
       
    if Choose == "Linear":
       #Linear trend for hwet:
       #h wet clay
       yC = mC*t + cC
       #h wet Sandy clay
       ySc = mSc*t + cSc
       #h wet Peat
       yP = mP*t + cP

    elif Choose == "Sin":
        #Sin trend for hwet:
        #h wet Sandy clay
        if hoSc != hwSc[1]:
           ySc=mSc * t +cSc + 0.2 * np.sin(t-np.pi)
        else:
            ySc=mSc * t +cSc
        #h wet clay
        if hoC != hwC[1]:
           yC = mC*t + cC + 0.2 * np.sin(t-np.pi)
        else:
            yC = mC*t + cC 
        #h wet Peat
        if hoP != hwP[1]:
           yP = mP*t + cP + 0.2 * np.sin(t-np.pi)
        else:
            yP = mP*t + cP
    
    return yC,yP,ySc

def ExtractData():
    
    # load the JSON data from file
    with open('Input.json', 'r') as f:
         data = json.load(f)

    # extract the value of a variable from the JSON data
    t = data['Time']
    mum = data['mum']
    devstm = data['devstm']
    mud = data['mud']
    devstd = data['devstd']
    Ncm = data['Ncm']
    Ncd = data['Ncd']
    hoP = data['hoP']
    hoSc = data['hoSc']
    hoC = data['hoC']
    hwSc = data['hwSc']
    hwP = data['hwP']
    hwC = data['hwC']
    N3 = data['N3']
    num_steps3= data['num_steps3']
    delta3 = data['delta3']
    x = data['x']
    Choose = data['Choose']

  
    return np.transpose(np.array(t)),np.array(mum).T,np.array(devstm).T,np.array(mud).T,np.array(devstd).T,Ncm,Ncd,hoP,hoSc,hoC,np.array(hwSc).T,np.array(hwP).T,np.array(hwC).T,N3,num_steps3,delta3,np.array(x).T,Choose


#=======================================================================================
#=======================================================================================
#=======================================================================================
#=================Main==================================================================
#=======================================================================================
#=======================================================================================

def main():
    
    #Further investigations:
    # diagonal covariance matrice neglet informations and indipedence of sampling

    #==========================================================
    #============Estract data from JSON======================== 
    #==========================================================
    t,mum,devstm,mud,devstd,Ncm,Ncd,hoP,hoSc,hoC,hwSc,hwP,hwC,N3,num_steps3,delta3,x,Choose = ExtractData()


    #==========================================================
    #================Groundwaterlevel========================== 
    #==========================================================
    yC,yP,ySc = Groundwaterlevel(t,hwSc,hwP,hwC,Choose,hoP,hoSc,hoC)
 

    #==========================================================
    #==============Covariance matrix========================== 
    #==========================================================
    covd,covm =CovarianceMatrix(mum,devstm,Ncm,Ncd,mud,devstd)


    #==========================================================
    #=================Metropolis=============================== 
    #==========================================================
    samples3,Prob3 = metropolis(N3, num_steps3, delta3, mum, covm, mud, covd, x,t, hoP, hoC, hoSc, yP, yC, ySc)

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




