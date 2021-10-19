# Copyright 2021, Blue Brain Project, EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
FROM nvidia/cuda:10.2-runtime

ENV http_proxy="http://bbpproxy.epfl.ch:80/"
ENV https_proxy="http://bbpproxy.epfl.ch:80/"
ENV LANG=C.UTF-8

# Install system packages and set up python
RUN DEBIAN_FRONTEND="noninteractive" \
apt-get update && \
apt-get install -y --no-install-recommends \
curl git htop less man vim \
libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx \
python3.7 python3.7-venv python3-pip &&\
update-alternatives --install /usr/local/bin/python python /usr/bin/python3.7 0 &&\
update-alternatives --install /usr/local/bin/python3 python3 /usr/bin/python3.7 0 &&\
python -m pip install -U pip wheel setuptools

# Install atlannot requirements, PyTorch and Jupyter
COPY requirements.txt /tmp
COPY requirements-dev.txt /tmp
COPY requirements-interactive.txt /tmp
RUN pip install --no-cache-dir \
-r /tmp/requirements.txt \
-r /tmp/requirements-dev.txt \
-r /tmp/requirements-interactive.txt \
torch torchvision jupyterlab ipywidgets

# Add and configure users
SHELL ["/bin/bash", "-c"]
ARG BBP_USERS
COPY docker/utils.sh /tmp
RUN \
. /tmp/utils.sh && \
groupadd -g 999 docker && \
create_users "guest/1000,${BBP_USERS}" "docker" && \
configure_user


# Entrypoint and working environment
EXPOSE 8888
RUN mkdir /workdir && chmod a+rwX /workdir
WORKDIR /workdir
USER guest
ENTRYPOINT ["env"]
CMD ["bash", "-l"]
