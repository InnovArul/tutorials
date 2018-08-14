Steps for Pytorch installation
------------------------------

1) Install Anaconda :
Goto Anaconda download page (https://www.anaconda.com/download/) and find a suitable version for your system.

cd some_dir
curl -O download_link
eg:curl -O https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh

bash Anaconda3-5.1.0-Linux-x86_64.sh

follow the instructions on screen...

--------------------------------------------
If it doesn't work, Install from source:

https://github.com/pytorch/pytorch#from-source

---------------------------------------------------------------------------------
2) Create Anaconda Environment for Pytorch

conda create --name my_env python=3.6

---------------------------------------------------------------------------------

3) Activate the environment

source activate my_env

---------------------------------------------------------------------------------
4) Install Pytorch : Goto https://pytorch.org and choose the suitable version for your system. 
The website will give the command to run.
eg: conda install pytorch torchvision -c pytorch

Done..!!
