### This is the code write up for Part 1 (Linear Model)

##import packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os




class linear_class():
    
    def __init__(self,location):
        self.df=pd.DataFrame()
        #variables used later for normalization 
        self.x_mean=0
        self.x_std=0
        self.y_mean=0
        self.y_std=0
        self.parameter=[]
        self.location=location
        self.iterations=[]; self.loss_train=[];  self.loss_val=[]        
        self.iteration=0
        
    def read(self):
        self.df=pd.read_json(self.location)
        self.df=self.df[['x','y']]
        self.df.columns=['age','weight']
        self.df=self.df.loc[self.df.age<18]
        
    def datasplit(self):##this function defines normalization and the split of training and testing data (0.8, 0.2)
        mean=self.df.mean()
        std=self.df.std()
        #normalization
#         self.df=(self.df-mean)/std
        self.x_mean=mean['age']
        self.x_std=std['age']
        self.y_mean=mean['weight']
        self.y_std=std['weight']
        n=len(self.df)
        index=np.random.rand(len(self.df))<0.8
        training=self.df[index]
        vali=self.df[~index]
        
        #to_numpy transforms data to array and saves computation time 
        self.train_x=training['age'].to_numpy()
        self.train_y=training['weight'].to_numpy()
        self.val_x=vali['age'].to_numpy()
        self.val_y=vali['weight'].to_numpy()
    
    def loss(self,p):##define loss function 
    
        training_loss=((self.train_y-self.model(self.train_x,p))**2).mean()
        val_loss=((self.val_y-self.model(self.val_x,p))**2).mean()
        self.loss_train.append(training_loss)
        self.loss_val.append(val_loss)
        self.iterations.append(self.iteration)      
        self.iteration+=1
                
        return training_loss
    
    def model(self,x,p):
        return p[0]+p[1]*x ##linear model 
                

    def model_optimize(self):
        po=np.random.uniform(0.5,1.0,size=2)
        #train model using scipy optimizer

        res =minimize(self.loss,po, method='Nelder-Mead',tol=1e-15)
        
        self.parameter=res.x
        print('OPTIMAL PARAM:', self.parameter)
    
    def visualization(self):
        
        # Show loss figure
        fig, ax = plt.subplots()       
        ax.plot(self.iterations  , self.loss_train  , 'ro', label="Training loss") 
        ax.plot(self.iterations  , self.loss_val  , 'bo', label="Validation loss")        
        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel('optimizer iterations', fontsize=FS)
        plt.ylabel('loss', fontsize=FS)
        plt.title('loss curve')
        plt.show()         
             
        fig, ax = plt.subplots()
        ax.plot(self.train_x, self.train_y, 'bo', label="Training set")
        ax.plot(self.val_x, self.val_y, 'yx', label="Validation set")
        ax.plot(self.df['age'].to_numpy(), self.model(self.df['age'].to_numpy(), self.parameter), 'r-', label="linear regression Model")    

        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel('age', fontsize=FS)
        plt.ylabel('weight', fontsize=FS)
        plt.title('linear regression')
        plt.show()      
       

if __name__=='__main__':
        
    location='/Users/jamesgao/590/590-CODES/DATA/weight.json'
     
    #linear regression
    lr=linear_class(location)
    lr.read()
    lr.datasplit()
    lr.model_optimize()
    lr.visualization()