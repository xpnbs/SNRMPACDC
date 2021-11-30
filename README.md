# SNRMPACDC
1. System requirements

   Hardware requirements:
   
   getparameters.py requires a computer with enough RAM to support the in-memory operations. 

   Operating system: windows 10
   
   Code dependencies: 
    - python 3.6
    - pandas '1.1.5'
    - numpy '1.18.5'
    - sklearn
    - tensorflow-gpu '1.13.1'
    - keras '2.2.4'
    - xlrd '1.2.0'
    -h5py '2.10.0'

2. Installation guide

    First, install CUDA 10.0.0 and CUDNN 7.6.4
    
    Second, install Anaconda3. Please refer to https://www.anaconda.com/distribution/ to install Anaconda 3.
    
    Third, open Anaconda Prompt to create a virtual environment by the following command:
    
    conda env create -n env_name -f environment.yaml
    
    Note: the environment.yaml file should be downloaded and put into the default path of Anaconda Prompt.


3. Demo
   Instructions to run on data:
   
   First, put the getparameter.py, test.py, cellline_expression.csv, cell-line-name.xlsx, copy number and mutation.csv, drugfeature.csv, drug-name.xlsx, label.csv into the same folder. 
   
   Second, open Anaconda Prompt. 
   
   Third, enter the following command:
   
   activate env_name
   
   Fourth, enter the following command:
   
   spyder
   
   Fifth, the spyder would be opend. Open the getparameter.py by spyder and run.
   
   Sixth, open the test.py by spyder and run.

  
   Expected output：
   
   The predcition result of new drug combinations for different cell lines would be output as different csv files.

   Expected run time for demo on a "normal" desktop computer:
   
   The run time in our coumputer (CPU:Xeon 3106, GPU NVIDIA Geforce RTX 2080 Ti, ARM 64G) is about one hour.

4. Instructions for use

   The real data is the same as the date in Demo. Therefore, instructions to run on our date is the same with the instructions to run on demo data.
