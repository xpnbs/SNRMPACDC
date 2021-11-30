import pandas as pd
import numpy as np
from scipy.special import comb
import getparameter
from keras import backend as K
celllinename=pd.read_excel("cell-line-name.xlsx",header=None)

drugname=pd.read_excel("drug-name.xlsx",header=None)
drug = pd.read_csv("drugfeature.csv",header=None)
cellline1 = pd.read_csv("cellline_expression.csv",header=None)
cellline2 = pd.read_csv("copy number and mutation.csv").T
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



#normalize+tanh+normalize
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


labels = pd.read_csv("label.csv",header=None)


labels = labels.values                                                                              


class_models = getparameter.build_model()
best_weights_filepath = './best_weights.hdf5'
class_models.load_weights(best_weights_filepath)


threshold=1.9
for lll in range(1):
    threshold += 0.1
    for i in range(31):   
        
        labelindex=np.where((labels[:,2]==(i+1))&(labels[:,3]>threshold))   
        
        labelindex=np.array(labelindex).reshape(-1,)
        

        olddrug1=labels[labelindex,0]
        olddrug2=labels[labelindex,1]
        olddrug=set(olddrug1) |set(olddrug2)
        
        b=[]   
        for j in range(38):
            b.append(j+1)
            newdrug = list((olddrug | set(b) ) - (olddrug & set(b)))
            
        olddrug=list(map(int,olddrug))
            
        drug1test=np.zeros((len(newdrug)*len(olddrug),2431),dtype=float)  
        drug2test=np.zeros((len(newdrug)*len(olddrug),2431),dtype=float) 
        celllinetest=np.zeros((len(newdrug)*len(olddrug),29813),dtype=float) 
        number=np.zeros((len(newdrug)*len(olddrug),2),dtype=float) 
        for m in range(len(olddrug)):
            for n in range(len(newdrug)):
                numberrrr=[]
                numberrrr.append(olddrug[m])
                numberrrr.append(newdrug[n])
                number[m*len(newdrug)+n]=numberrrr
                
                drug111=[]
                drug222=[]
                celllineeee=[]
                drug111.extend(drug[olddrug[m]-1])     
                drug222.extend(drug[newdrug[n]-1])
                celllineeee.extend(cellline[i])
                drug1test[m*len(newdrug)+n]=drug111
                drug2test[m*len(newdrug)+n]=drug222
                celllinetest[m*len(newdrug)+n]=celllineeee

        labelfuindex=[]
        labelfuindex1=[]
        labelfuindex2=[]
 
        labelscellline=labels[labels[:,2]==(i+1)]
        for q in range(len(labelscellline)):
            labelfuindex += np.array(np.where((labelscellline[q][0]==number[:,0])&(labelscellline[q][1]==number[:,1]))).reshape(-1,).tolist()
            labelfuindex1 += np.array(np.where((labelscellline[q][0]==number[:,1])&(labelscellline[q][1]==number[:,0]))).reshape(-1,).tolist()
    
        labelfuindex2=labelfuindex+labelfuindex1
        labelfuindex2=list(set(labelfuindex2))

        drug1test=np.delete(drug1test,labelfuindex2,axis=0)
        drug2test=np.delete(drug2test,labelfuindex2,axis=0)
        celllinetest=np.delete(celllinetest,labelfuindex2,axis=0)
        number=np.delete(number,labelfuindex2,axis=0)
        
        if(len(newdrug)>1):
            zuhe=int(comb(len(newdrug),2))
            drug1xinxin=np.zeros((zuhe,2431),dtype=float)  
            drug2xinxin=np.zeros((zuhe,2431),dtype=float) 
            celllinexinxin=np.zeros((zuhe,29813),dtype=float) 
            numberxinxin=np.zeros((zuhe,2),dtype=float) 
    
            count=-1
            for z in range(len(newdrug)):
                for x in range(z+1,len(newdrug)):
                    count+=1
                    numberrrr=[]
                    numberrrr.append(newdrug[z])
                    numberrrr.append(newdrug[x])
                    numberxinxin[count]=numberrrr
                    
                    drug111=[]
                    drug222=[]
                    celllineeee=[]
                    drug111.extend(drug[newdrug[z]-1])     
                    drug222.extend(drug[newdrug[x]-1])
                    celllineeee.extend(cellline[i])
                    drug1xinxin[count]=drug111
                    drug2xinxin[count]=drug222
                    celllinexinxin[count]=celllineeee
        
        drug1test = np.reshape(drug1test, (-1, 2431,1))
        drug2test = np.reshape(drug2test, (-1, 2431,1))
        celllinetest = np.reshape(celllinetest, (-1, 29813,1))
        
        if(len(celllinetest)>0):
            scorelaoxin=class_models.predict([celllinetest,drug1test,drug2test])
            scorelaoxin=scorelaoxin.reshape(-1,1)
            
            results=np.hstack((number,scorelaoxin))
            results=results[np.argsort(-results[:,2])]
            results=pd.DataFrame(results)
            
            list1 = []
            list2=[]
            for ii in range(len(results)):
            
                aaa=results.loc[ii][0]
                list1.append(drugname.loc[aaa-1][1])
                
                aaaaa=results.loc[ii][1]
                list2.append(drugname.loc[aaaaa-1][1])
            
            list1 = pd.DataFrame(list1)
            list2 = pd.DataFrame(list2)
    
            results=pd.concat((list1,list2,results),axis=1)
            results.columns = ['2e','hei','tyt','hbh','jhg']
            results.drop(['tyt','hbh'],axis=1,inplace = True)
        
            results.to_csv(celllinename.loc[i][1]+' '+'old drug_new drug'+'.csv',header=None,index=False)    
        if(len(newdrug)>1):
            drug1xinxin = np.reshape(drug1xinxin, (-1, 2431,1))
            drug2xinxin = np.reshape(drug2xinxin, (-1, 2431,1))
            celllinexinxin = np.reshape(celllinexinxin, (-1, 29813,1))
            scorexinxin=class_models.predict([celllinexinxin,drug1xinxin,drug2xinxin])
            scorexinxin=scorexinxin.reshape(-1,1)
            resultsxinxin=np.hstack((numberxinxin,scorexinxin))
            resultsxinxin=pd.DataFrame(resultsxinxin)
            
            list3 = []
            list4=[]
            for ii in range(len(resultsxinxin)):
            
                aaa=resultsxinxin.loc[ii][0]
                list3.append(drugname.loc[aaa-1][1])
                
                aaaaa=resultsxinxin.loc[ii][1]
                list4.append(drugname.loc[aaaaa-1][1])
            
            list3 = pd.DataFrame(list3)
            list4 = pd.DataFrame(list4)
    
            resultsxinxin=pd.concat((list3,list4,resultsxinxin),axis=1)
            resultsxinxin.columns = ['2e','hei','tyt','hbh','jhg']
            resultsxinxin.drop(['tyt','hbh'],axis=1,inplace = True)
            
            
            resultsxinxin.to_csv(celllinename.loc[i][1]+' '+'new drug_new drug'+'.csv',header=None,index=False)  
