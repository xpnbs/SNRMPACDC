import pandas as pd
import numpy as np
from keras.models import Sequential 
from keras.layers import merge, Conv2D, MaxPool2D, Activation, Dense, concatenate, Flatten,Multiply,Lambda,Reshape,BatchNormalization,GlobalAveragePooling2D,Conv1D,MaxPool1D,GlobalAveragePooling1D,Dropout
from keras.layers import Input 
from keras.models import Model 
from keras.utils import np_utils 
import tensorflow as tf 
import keras 
from keras.datasets import mnist 
import numpy as np 
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau 
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.optimizers import Adam
from keras import regularizers,callbacks
from sklearn import metrics
import math

def build_model():

    inp3=Input(shape=(29813,1)) 
    
    models1 = Conv1D(16,3,padding='same')(inp3) 
    models1=BatchNormalization()(models1)
    models1 = Activation('relu')(models1) 
    models1 = MaxPool1D(2,2,padding='same')(models1) 
    
    models1 = Conv1D(16,3,padding='same')(models1) 
    models1=BatchNormalization()(models1)
    models1 = Activation('relu')(models1) 
    models1 = MaxPool1D(2,2,padding='same')(models1)
    
    models1 = Conv1D(16,3,padding='same')(models1) 
    models1=BatchNormalization()(models1)
    models1 = Activation('relu')(models1) 
    models1 = MaxPool1D(2,2,padding='same')(models1)
    
    models1 = Conv1D(16,3,padding='same')(models1) 
    models1=BatchNormalization()(models1)
    models1 = Activation('relu')(models1) 
    models1 = MaxPool1D(2,2,padding='same')(models1)
    
    models1 = Conv1D(16,3,padding='same')(models1) 
    models1=BatchNormalization()(models1)
    models1 = Activation('relu')(models1) 
    models1 = MaxPool1D(2,2,padding='same')(models1) 
        
    models1=Flatten()(models1)
    models1=BatchNormalization()(models1)
    models1 = Activation('relu')(models1)
    models1=Dropout(0.2)(models1)
    
    models1 = Dense(4096)(models1)    
    model1 = Model(inputs=inp3, outputs=models1) 

    inp3 = Input(shape=(29813,1))
    model_3 = model1(inp3)
    
    inp=Input(shape=(2431,1), name='FeatureNet_ImageInput') 
    
    models2 = Conv1D(64,3,padding='same')(inp) 
    models2=BatchNormalization()(models2)
    models2 = Activation('relu')(models2) 
    models2 = MaxPool1D(2,2,padding='same')(models2) 
    
    models2 = Conv1D(64,3,padding='same')(models2) 
    models2=BatchNormalization()(models2)
    models2 = Activation('relu')(models2) 
    models2 = MaxPool1D(2,2,padding='same')(models2)
    
    models2 = Conv1D(64,3,padding='same')(models2) 
    models2=BatchNormalization()(models2)
    models2 = Activation('relu')(models2) 
    models2 = MaxPool1D(2,2,padding='same')(models2)
    
    models2 = Conv1D(64,3,padding='same')(models2) 
    models2=BatchNormalization()(models2)
    models2 = Activation('relu')(models2) 
    models2 = MaxPool1D(2,2,padding='same')(models2)
    
    
    models2 = Conv1D(64,3,padding='same')(models2) 
    models2=BatchNormalization()(models2)
    models2 = Activation('relu')(models2) 
    models2 = MaxPool1D(2,2,padding='same')(models2) 
    

    models2=Flatten()(models2)
    models2=BatchNormalization()(models2)
    models2 = Activation('relu')(models2)
    models2=Dropout(0.2)(models2)
    
    models2 = Dense(4096)(models2) 

    
    
    models2=Reshape([1,4096])(models2)
    model = Model(inputs=inp, outputs=models2) 

    
    inp1 = Input(shape=(2431 ,1)) 
    inp2 = Input(shape=(2431,1)) 
    model_1 = model(inp1)
    model_2 = model(inp2) 

    A=np.random.randn(2048,4096)
    B=np.random.randn(2048,4096)
    AA=np.vstack((A,B))
    BB=np.vstack((B,A))
    
    AA=K.variable(value=AA,dtype='float32',name='LALA')
    BB=K.variable(value=BB,dtype='float32',name='LALA')
    
    model_11=Lambda(lambda x:K.dot(x,AA))(model_1)
    model_22=Lambda(lambda x:K.dot(x,BB))(model_2)
    

    merge_layers = Multiply()([model_11,model_22])
    merge_layers=Reshape([4096])(merge_layers)
       
   
    merge_layers = Multiply()([model_3, merge_layers])
    
    merge_layers=BatchNormalization()(merge_layers) 
    merge_layers = Activation('relu')(merge_layers)
    merge_layers=Dropout(0.5)(merge_layers)
    merge_layers = Dense(2048)(merge_layers)
    merge_layers=BatchNormalization()(merge_layers) 
    merge_layers = Activation('relu')(merge_layers)
    merge_layers=Dropout(0.5)(merge_layers)

    merge_layer=Dense(1,kernel_regularizer=regularizers.l2(0.0001))(merge_layers)

    class_models=Model(inputs=[inp3,inp1, inp2],outputs=merge_layer)

    return class_models
if __name__=="__main__":
    drug = pd.read_csv("drugfeature.csv",header=None)
    cellline1 = pd.read_csv("cellline_expression.csv",header=None) 
    cellline2 = pd.read_csv("copy number and mutation.csv").T  
    labels = pd.read_csv("label.csv",header=None)

    labels = labels.values                                                                              
    cell_line=cellline1.values
    cellline2=cellline2.values
       
    std_drug = np.nanstd(drug, axis=0) 
    feat_filt = std_drug!=0
    drug=drug.values
    drug = drug[:,feat_filt]
    
    std_cellline2 = np.nanstd(cellline2, axis=0)
    feat_filt11 = std_cellline2!=0
    cellline2 = cellline2[:,feat_filt11]
    
    std_cellline = np.nanstd(cell_line, axis=0)
    feat_filt1 = std_cellline!=0
    cell_line = cell_line[:,feat_filt1]
    
    mean_drug1_1 = np.mean(drug, axis=0)
    
    mean_cell_line_1 = np.mean(cell_line, axis=0)
    drug = (drug-mean_drug1_1)/std_drug[feat_filt]
    cell_line = (cell_line-mean_cell_line_1)/std_cellline[feat_filt1]
    
    drug = np.tanh(drug)
    cell_line = np.tanh(cell_line)
    
    mean_drug1_2 = np.mean(drug, axis=0)
    mean_cell_line_2 = np.mean(cell_line, axis=0)
    std_drug_2 = np.std(drug, axis=0)
    std_cellline_2 = np.std(cell_line, axis=0)
    drug = (drug-mean_drug1_2)/std_drug_2
    cell_line = (cell_line-mean_cell_line_2)/std_cellline_2
    drug[:,std_drug_2==0]=0
    
    cellline=np.hstack((cell_line,cellline2))
    
    
    del (cellline1,feat_filt,feat_filt1,mean_drug1_1,mean_drug1_2,mean_cell_line_1,mean_cell_line_2,
         std_cellline,std_cellline_2,std_drug,std_drug_2)
    
    drug1=np.zeros((18283,2431))
    drug2=np.zeros((18283,2431))
    cell_line_a = np.zeros((18283,29813))
    label = np.zeros((18283,1))
    for i in range(18283):
        drug1[i] = drug[int(labels[i][0])-1]
        drug2[i] = drug[int(labels[i][1])-1]
        cell_line_a[i] = cellline[int(labels[i][2])-1]
        label[i]=labels[i][3]
        
    drug11=np.vstack((drug1,drug2))
    drug22=np.vstack((drug2,drug1))
    cell_line_aa=np.vstack((cell_line_a,cell_line_a))
    labell=np.vstack((label,label))

    K.clear_session()

    drug11,drug22,cell_line_aa,labell=shuffle(drug11,drug22,cell_line_aa,labell)
    
    drug11 = np.reshape(drug11, (-1, 2431,1))
    drug22 = np.reshape(drug22, (-1, 2431,1))
    cell_line_aa = np.reshape(cell_line_aa, (-1, 29813,1))
    
    class_models = build_model()
         
    adam=Adam(lr=1e-4)
    
    class_models.compile(optimizer=adam, loss='mse')
    best_weights_filepath = './best_weights.hdf5'
    earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')
    saveBestModel = callbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='auto')
    
    # train model
    history = class_models.fit([cell_line_aa,drug11, drug22], labell, batch_size=128, nb_epoch=50,verbose=1,validation_split=0.2, callbacks=[earlyStopping, saveBestModel],shuffle=True)


    
    


    




