
##import packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os




class logistic_second_class():
    
    def __init__(self,location):
        self.df=pd.DataFrame()
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
        self.df=self.df[['y','is_adult']]
        self.df.columns=['weight','is_adult']
    
    def datasplit(self):
        mean=self.df.mean()
        std=self.df.std()
        #normalization
        self.df=(self.df-mean)/std
        self.x_mean=mean['weight']
        self.x_std=std['weight']
        self.y_mean=mean['is_adult']
        self.y_std=std['is_adult']
        n=len(self.df)
        index=np.random.rand(len(self.df))<0.8
        training=self.df[index]
        vali=self.df[~index]
        
        self.train_x=training['weight'].to_numpy()
        self.train_y=training['is_adult'].to_numpy()
        self.val_x=vali['weight'].to_numpy()
        self.val_y=vali['is_adult'].to_numpy()
    
    def loss(self,p):
    
        training_loss=((self.train_y-self.model(self.train_x,p))**2).mean()
        val_loss=((self.val_y-self.model(self.val_x,p))**2).mean()
        self.loss_train.append(training_loss)
        self.loss_val.append(val_loss)
        self.iterations.append(self.iteration)      
        self.iteration+=1
                
        return training_loss
    
    def model(self,x,p):##define another logistic model 
        return p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))
                

    def model_optimize(self):
        np.random.seed(0)
        po=np.random.uniform(0.5,1.0,size=4)
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
        ax.plot(self.train_x*self.x_std+self.x_mean, self.train_y*self.y_std+self.y_mean, 'bo', label="Training set")
        ax.plot(self.val_x*self.x_std+self.x_mean, self.val_y*self.y_std+self.y_mean, 'yx', label="Validation set")
        x=self.df['weight'].to_numpy()*self.x_std+self.x_mean
        y_model=self.model(self.df['weight'],self.parameter).to_numpy()*self.y_std+self.y_mean
        y_model=np.where(y_model>0.5,1,0)
        x=x.tolist()
        y_model=y_model.tolist()
        new_x,new_y_model=zip(*sorted(zip(x,y_model)))
        ax.plot(new_x, new_y_model, 'r-', label="logistic regression Model") 
        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel('weight', fontsize=FS)
        plt.ylabel('is_adult', fontsize=FS)
        plt.title('Logistic regression')
        plt.show()      
       

if __name__=='__main__':
        
    location='/Users/jamesgao/590/590-CODES/DATA/weight.json'
     
    #linear regression
    lr=logistic_second_class(location)
    lr.read()
    lr.datasplit()
    lr.model_optimize()
    lr.visualization()