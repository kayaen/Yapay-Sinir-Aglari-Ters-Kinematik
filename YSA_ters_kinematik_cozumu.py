# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:31:28 2019

@author: engin

drawing arm taken from:
http://enesbot.me/tutorial-forward-kinematics-part-i.html

Ileri beslemeli sinir aglariyla ters kinematik cozumu
"""

import numpy as np
import matplotlib.pyplot as plt
import keras

def donusumMatrisi(DH_satir):
    alpha, a, d, theta = DH_satir[0,0], DH_satir[0,1], DH_satir[0,2], DH_satir[0,3]
    T = np.matrix ([[np.cos(theta)             , -np.sin(theta)              ,  0            ,  a],
                   [np.sin(theta)*np.cos(alpha),  np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                   [np.sin(theta)*np.sin(alpha),  np.cos(theta)*np.sin(alpha),  np.cos(alpha),  np.cos(alpha)*d],
                   [0                          ,  0                          ,  0            ,  1]])
    return T

def ileriKinematik(DH_Matrisi):
    T01 = donusumMatrisi(DH_Matrisi[0,:])
    T12 = donusumMatrisi(DH_Matrisi[1,:])
    T23 = donusumMatrisi(DH_Matrisi[2,:])
    T3E = donusumMatrisi(DH_Matrisi[3,:])    
    T1 = T01
    T2 = T1*T12
    T3 = T2*T23
    T0E = T01*T12*T23*T3E
    return T0E, T3, T2, T1

def konumVeAci(T):
    return(T[0,3],T[1,3],np.arctan2(T[1,0],T[0,0]))


def tahmin_ciz(TE,T3,T2,T1):
    plt.clf()

    plt.plot(T1[0,3],T1[1,3], marker='o', linestyle='--', color='k')
    plt.plot(T2[0,3],T2[1,3], marker='o', linestyle='--', color='k')
    plt.plot(T3[0,3],T3[1,3], marker='o', linestyle='--', color='k')

    toolp1 = np.matrix([[0],[0.5],[0]]) 
    toolp2 = np.matrix([[0],[-0.5],[0]]) 
    toolp3 = np.matrix([[0.25],[0.5],[0]]) 
    toolp4 = np.matrix([[0.25],[-0.5],[0]]) 

    toolp1 = T3[:3,:3]*toolp1 + TE[0:3,3]
    toolp2 = T3[:3,:3]*toolp2 + TE[0:3,3]
    toolp3 = T3[:3,:3]*toolp3 + TE[0:3,3]
    toolp4 = T3[:3,:3]*toolp4 + TE[0:3,3]    

    plt.plot([toolp1[0,0], toolp2[0,0]], [toolp1[1,0], toolp2[1,0]], color='k', linewidth=2)
    plt.plot([toolp1[0,0], toolp3[0,0]], [toolp1[1,0], toolp3[1,0]], color='k', linewidth=2)
    plt.plot([toolp4[0,0], toolp2[0,0]], [toolp4[1,0], toolp2[1,0]], color='k', linewidth=2)

    plt.plot([0,T2[0,3]],       [0, T2[1,3]],       color='k', linewidth=2)
    plt.plot([T2[0,3],T3[0,3]], [T2[1,3],T3[1,3]],  color='k', linewidth=2)    
    plt.plot([T3[0,3],TE[0,3]], [T3[1,3],TE[1,3]],  color='k', linewidth=2)    
    
    plt.xlim([-1,4])
    plt.ylim([-1,4])    


def hedef_ciz(TE,T3,T2,T1):
    plt.plot(T1[0,3],T1[1,3], marker='o', linestyle='--', color='r')
    plt.plot(T2[0,3],T2[1,3], marker='o', linestyle='--', color='r')
    plt.plot(T3[0,3],T3[1,3], marker='o', linestyle='--', color='r')

    toolp1 = np.matrix([[0],[0.5],[0]]) 
    toolp2 = np.matrix([[0],[-0.5],[0]]) 
    toolp3 = np.matrix([[0.25],[0.5],[0]]) 
    toolp4 = np.matrix([[0.25],[-0.5],[0]]) 

    toolp1 = T3[:3,:3]*toolp1 + TE[0:3,3]
    toolp2 = T3[:3,:3]*toolp2 + TE[0:3,3]
    toolp3 = T3[:3,:3]*toolp3 + TE[0:3,3]
    toolp4 = T3[:3,:3]*toolp4 + TE[0:3,3]    

    plt.plot([toolp1[0,0], toolp2[0,0]], [toolp1[1,0], toolp2[1,0]], color='r', linewidth=2)
    plt.plot([toolp1[0,0], toolp3[0,0]], [toolp1[1,0], toolp3[1,0]], color='r', linewidth=2)
    plt.plot([toolp4[0,0], toolp2[0,0]], [toolp4[1,0], toolp2[1,0]], color='r', linewidth=2)

    plt.plot([0,T2[0,3]],       [0, T2[1,3]],       color='r', linewidth=2)
    plt.plot([T2[0,3],T3[0,3]], [T2[1,3],T3[1,3]],  color='r', linewidth=2)    
    plt.plot([T3[0,3],TE[0,3]], [T3[1,3],TE[1,3]],  color='r', linewidth=2)    
    
    plt.xlim([-1,4])
    plt.ylim([-1,4])    

    plt.show()
    plt.pause(0.05)


def veriToplama(ornekSayisi):
    theta1_vec, theta2_vec, theta3_vec = [], [], []
    x_vec, y_vec, aci_vec = [], [], []
    aciAraligi = [0,np.pi/2]

    for i in range(ornekSayisi):
        while True:
            theta1 = np.random.uniform(aciAraligi[0], aciAraligi[1])
            theta2 = np.random.uniform(aciAraligi[0], aciAraligi[1])
            theta3 = np.random.uniform(aciAraligi[0], aciAraligi[1])
            DH_Matrisi = np.matrix([[0,0,0,theta1],
                                    [0,1,0,theta2],
                                    [0,1,0,theta3],
                                    [0,1,0,0]])
            T0E,_,_,_ = ileriKinematik(DH_Matrisi)            
            x, y, theta = konumVeAci(T0E)
            if x > 0 and y > 0:
                break

        theta1_vec.append(theta1)
        theta2_vec.append(theta2)
        theta3_vec.append(theta3)
        x_vec.append(x)
        y_vec.append(y)
        aci_vec.append(theta)
        
    veriMatrisi = np.c_[theta1_vec, theta2_vec, theta3_vec, x_vec, y_vec, aci_vec]
    return veriMatrisi

    
ornekSayisi = 2000
veriMatrisi = veriToplama(ornekSayisi)

plt.figure(0)
for i in range(veriMatrisi.shape[0]):
    plt.plot([veriMatrisi[i, 3], veriMatrisi[i, 3]+0.1*np.cos(veriMatrisi[i, 5])], [veriMatrisi[i, 4], veriMatrisi[i, 4]+0.1*np.sin(veriMatrisi[i, 5])], 'k-')
    plt.scatter(veriMatrisi[i, 3], veriMatrisi[i, 4], c='blue')
plt.xlabel("X Ekseni")
plt.ylabel("Y Ekseni")
plt.title("Toplanan verin konum ve yönelimleri")
plt.show()
#plt.savefig('DataSetEndEffectorPositionsAndOrientations.png')

inputMat = veriMatrisi[:, 3:6] #  Q1,Q2,Q3
outputMat = veriMatrisi[:, 0:3]  # x, y, ang

# Separate data set in to Train, Test And Validation
egitim_input = inputMat[0:int(0.85*ornekSayisi), :]                             
egitim_output = outputMat[0:int(0.85*ornekSayisi), :]

test_input = inputMat[int(0.85*ornekSayisi):int(ornekSayisi),:]
test_output = outputMat[int(0.85*ornekSayisi):int(ornekSayisi),:]

# define model

model = keras.Sequential()
model.add(keras.layers.Dense(40, activation='relu', input_dim=3))
model.add(keras.layers.Dense(40, activation='relu'))
model.add(keras.layers.Dense(3, use_bias=True, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.summary()
history = model.fit(egitim_input, egitim_output, epochs=100, verbose=1)
# plt.plot(history.epoch, history.history['loss'])

[loss, acc] = model.evaluate(test_input, test_output, verbose=0)       

test_tahminler = model.predict(test_input)

plt.figure(1)
for i in range(10):
    DH_Matrisi = np.matrix([[0,0,0,test_tahminler[i,0]],
                            [0,1,0,test_tahminler[i,1]],
                            [0,1,0,test_tahminler[i,2]],
                            [0,1,0,0]])
    TE,_,_,_ = ileriKinematik(DH_Matrisi)    
    x_tahmin, y_tahmin, aci_tahmin = konumVeAci(TE)  
    plt.scatter(x_tahmin, y_tahmin, marker='o', color='black')                              
    plt.plot([x_tahmin, x_tahmin+0.2*np.cos(aci_tahmin)], [y_tahmin, y_tahmin+0.2*np.sin(aci_tahmin)], 'k-')
    
    x_hedef, y_hedef, aci_hedef = test_input[i,:] 
    plt.scatter(x_hedef, y_hedef, marker='*', color='green')                              
    plt.plot([x_hedef, x_hedef+0.3*np.cos(aci_hedef)], [y_hedef, y_hedef+0.3*np.sin(aci_hedef)], 'g-')
    
    euclidean_hata = np.linalg.norm([x_tahmin-x_hedef, y_tahmin-y_hedef])  # unit
    acisal_hata = np.rad2deg(np.abs(aci_tahmin-aci_hedef))         # degree
    print('Tahmin:%s'%([x_tahmin, y_tahmin, aci_tahmin]))
    print('Hedef:%s '%([x_hedef, y_hedef, aci_hedef]))
    print('euclidean_hata:%s unit --- acisal_hata:%s deg.'%(euclidean_hata,   acisal_hata))
plt.xlabel("X Ekseni")
plt.ylabel("Y Ekseni")
plt.title("Test sonuçları")    
plt.legend(['Tahminler','Hedefler'])
plt.show()
#plt.savefig('TestSonuclari.png')

plt.figure(2)
for i in range(10):
    DH_Matrisi = np.matrix([[0,0,0,test_tahminler[i,0]],
                            [0,1,0,test_tahminler[i,1]],
                            [0,1,0,test_tahminler[i,2]],
                            [0,1,0,0]])
    
    TE,T3,T2T,T1 = ileriKinematik(DH_Matrisi)
    tahmin_ciz(TE,T3,T2T,T1)   #

    DH_Matrisi = np.matrix([[0,0,0,test_output[i,0]],
                            [0,1,0,test_output[i,1]],
                            [0,1,0,test_output[i,2]],
                            [0,1,0,0]])    
    TE,T3,T2H,T1 = ileriKinematik(DH_Matrisi)
    hedef_ciz(TE,T3,T2H,T1)  # 
plt.xlabel("X Ekseni")
plt.ylabel("Y Ekseni")
plt.title("Tahmin ve Hedef Robot Kol Pozisyonları")   
plt.show()
#plt.savefig('RobotKolCizimler.png')










