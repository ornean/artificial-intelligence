conda init powershell
conda create -n DQN-LunarLander python=3.8 -y
conda activate DQN-LunarLander
conda install -c anaconda swig -y
conda install pytorch cudatoolkit=11.3 -c pytorch -y
conda install numpy -y
conda install -c conda-forge gym[all] -y
# pip install -r requirements.txt
conda env update --file environment.yml
