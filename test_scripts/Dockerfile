FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND="noninteractive"
ENV CUDA_HOME="/usr/local/cuda"

ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH}

SHELL ["/bin/bash", "-c"]

WORKDIR /root

# # set apt source to Tsinghua University mirror
# RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
#     echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
#     echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
#     echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list
# # Prevent the docker-clean script from deleting .deb files
# RUN sed -i 's/^DPkg::Post-Invoke/#DPkg::Post-Invoke/' /etc/apt/apt.conf.d/docker-clean && \
#     sed -i 's/^APT::Update::Post-Invoke/#APT::Update::Post-Invoke/' /etc/apt/apt.conf.d/docker-clean

# set timezone to Asia/Shanghai
RUN \
        apt update && \
        apt-get install -y tzdata && \
        ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
        dpkg-reconfigure --frontend noninteractive tzdata

RUN \
        apt install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip screen \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-dev locales \
        p7zip-full zip inotify-tools cron lsof git git-lfs && \
        git lfs install && \
        locale-gen en_US.UTF-8


# node.js
RUN \
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash && \
        . "/root/.nvm/nvm.sh" && \
        nvm install 22

RUN \
        cd /root && \
        apt remove python3-blinker -y && \
        pip install --break-system-packages -U blinker && \
        pip install nltk regex six pybind11 openpyxl transformers websockets flask-restful && \
        pip install datasets flake8 huggingface-hub[cli,hf_xet] sentencepiece pandas h5py matplotlib psutil tqdm tiktoken seaborn \
        scikit-learn spacy pillow scipy tensorboard wandb pyyaml ipython ipdb ipykernel ipywidgets && \
        python -m nltk.downloader punkt && \
        python -m nltk.downloader wordnet

RUN \
        git clone https://github.com/OpenSQZ/MegatronApp.git && \
        cd MegatronApp  && \
        bash prerequisite.sh  && \
        cd megatron/shm_tensor_new_rdma  && \
        pip install -e . && \
        cd ../.. && \
        cd megatron/shm_tensor_new_rdma_pre_alloc  && \
        pip install -e . && \
        cd ../..


# SSH config
RUN \
        echo 'root:root' | chpasswd && \
        cp /etc/ssh/sshd_config /tmp/sshd_config && \
        echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config && \
        sed -i "s/#Port 22/Port 22/" /etc/ssh/sshd_config && \
        sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config && \
        sed -i "s/#PasswordAuthentication yes/PasswordAuthentication yes/" /etc/ssh/sshd_config && \
        sed -i "s/#PubkeyAuthentication yes/PubkeyAuthentication yes/" /etc/ssh/sshd_config && \
        sed -i "s/UsePAM yes/UsePAM no/" /etc/ssh/sshd_config && \
        sed -i "s/#StrictModes yes/StrictModes no/" /etc/ssh/sshd_config && \
        sed -i "s/#   StrictHostKeyChecking ask/    StrictHostKeyChecking no/" /etc/ssh/ssh_config && \
        chown root:root /etc/ssh/sshd_config && \
        mkdir -p /run/sshd && chmod 0755 /run/sshd && \
        ssh-keygen -t rsa -f /root/.ssh/id_rsa -q -N "" && \
        cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
# ENV config
RUN \
        echo "export CUDA_HOME=${CUDA_HOME}" | cat - ~/.bashrc > /tmp/.bashrc && mv /tmp/.bashrc ~/.bashrc && \
        echo "export PATH=$PATH" | cat - ~/.bashrc > /tmp/.bashrc && mv /tmp/.bashrc ~/.bashrc && \
        echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | cat - ~/.bashrc > /tmp/.bashrc && mv /tmp/.bashrc ~/.bashrc && \
        echo "export LIBRARY_PATH=$LIBRARY_PATH" | cat - ~/.bashrc > /tmp/.bashrc && mv /tmp/.bashrc ~/.bashrc

