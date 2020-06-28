FROM nvidia/cuda:9.2-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    apt-utils \
    curl \
    ca-certificates \
    sudo \
    bzip2 \
    libx11-6

# Create a working directory
RUN mkdir /workdir
WORKDIR /workdir
RUN chmod -R 777 /workdir 

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN chown -R user:user /workdir
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# Install Anaconda
RUN curl -so ~/anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
RUN chmod +x ~/anaconda.sh
RUN ~/anaconda.sh -b -p ~/conda
ENV PATH=/home/user/conda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Install dependencies
RUN conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
RUN pip install torchnet wtfml pretrainedmodels torchsummary albumentations
RUN conda install -y -c anaconda numpy pandas matplotlib scipy scikit-learn pillow joblib tqdm jupyter
RUN pip install timm tensorboardX hyperopt
RUN pip install efficientnet_pytorch

# will contain data, src and notebooks
RUN mkdir /workdir/data/
RUN mkdir /workdir/src/
RUN mkdir /workdir/notebooks/
RUN mkdir /workdir/models/
RUN mkdir /workdir/logs/

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
 
