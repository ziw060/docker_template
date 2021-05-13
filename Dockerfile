# 1) choose base container
# generally use the most recent tag

# data science notebook
# https://hub.docker.com/repository/docker/ucsdets/datascience-notebook/tags
# ARG BASE_CONTAINER=ucsdets/ucsdets/scipy-ml-notebook:2021.1-stable

# scipy/machine learning (tensorflow)
# https://hub.docker.com/repository/docker/ucsdets/scipy-ml-notebook/tags
ARG BASE_CONTAINER=nvidia/cuda:10.1-base-ubuntu16.04

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

RUN rm -rf /usr/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/bin/python3

RUN apt-get update && \
	apt-get install -y \
			libtinfo5 htop nvidia-cuda-toolkit python3-pip wget

RUN wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda

ENV PATH=/opt/conda/bin:$PATH

RUN conda install cudatoolkit=10.1 \
				  cudnn \
				  nccl \
				  -y

# Install pillow<7 due to dependency issue https://github.com/pytorch/vision/issues/1712
RUN pip install --no-cache-dir  datascience \
								PyQt5 \
								scapy \
								nltk \
								jupyter-tensorboard \
								pycocotools \
								"pillow<7" \
								tensorflow-gpu>=2.2 \ tensorflow-addons==0.11.2 \ 
                                tensorflow-datasets==4.3.0 \
                                tensorflow-estimator==2.2.0 \
                                tensorflow-metadata==0.30.0 
RUN pip install --no-cache-dir --upgrade jax \
    jaxlib==0.1.66+cuda101 -f \
    https://storage.googleapis.com/jax-releases/jax_releases.html


# 3) install packages
RUN pip install --no-cache-dir networkx scipy python-louvain

# 4) change back to notebook user
COPY /run_jupyter.sh /
RUN chmod 755 /run_jupyter.sh
USER $NB_UID

# Override command to disable running jupyter notebook at launch
# CMD ["/bin/bash"]
