conda init powershell
conda create -n branching-dqn python=3.8 -y

conda activate branching-dqn
conda install -c anaconda swig -y
#conda install pytorch cudatoolkit=11.3 -c pytorch -y
conda install numpy -y
conda install -c conda-forge gym[all] -y
pip install -r requirements.txt
