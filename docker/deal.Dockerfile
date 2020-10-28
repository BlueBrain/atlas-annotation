FROM nvcr.io/nvidia/tensorflow:20.09-tf1-py3

ARG BBP_HTTP_PROXY
ARG BBP_HTTPS_PROXY
ENV http_proxy=$BBP_HTTP_PROXY
ENV https_proxy=$BBP_HTTPS_PROXY

# Fix CV2 error: ImportError: libSM.so.6: cannot open shared object file: No such file or directory
RUN \
apt-get update && \
apt-get install -y libsm6 libxext6 libxrender-dev

# Install NodeJS
RUN \
curl -sL https://deb.nodesource.com/setup_14.x | bash - && \
apt-get install -y nodejs && \
update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip and co.
RUN pip install -U pip setuptools wheel

# Install jupyterlab extensions
RUN \
jupyter labextension install --no-build @jupyter-widgets/jupyterlab-manager && \
jupyter labextension install --no-build @jupyterlab/toc && \
jupyter-lab build --name "DEAL | DGX"

# Install Atlas Alignment
# RUN pip install git+https://bbpcode.epfl.ch/code/a/ml/atlas_alignment
COPY ./atlas_alignment /tmp/atlas_alignment
RUN pip install /tmp/atlas_alignment

# Install other packages
RUN pip install ipywidgets

# Add custom users specified in $BBS_USERS="user1/id1,user2/id2,etc"
ARG BBP_USERS
COPY ./docker/scripts.sh /tmp
RUN \
source /tmp/scripts.sh && \
groupadd -g 999 docker && \
create_users "$BBP_USERS" "docker" && \
add_aliases "/root" && \
improve_prompt "/root" "03" "36"

# Non-Root User
RUN useradd --create-home --uid 1000 --gid docker deal_user
USER deal_user

ENTRYPOINT ["bash"]

