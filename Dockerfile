FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel as BASE

RUN git clone https://github.com/nvidia/apex && \
    cd apex && \
    python setup.py install --user --cuda_ext --cpp_ext && \
    rm -rf /apex

FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
COPY --from=BASE /root/.local /root/.local

RUN cp /etc/apt/sources.list /etc/apt/sources.list~
RUN sed -Ei 's/^# deb-src /deb-src /' /etc/apt/sources.list
RUN apt-get update
RUN apt-get -y install software-properties-common
RUN apt-get update
RUN apt-get build-dep ffmpeg -y
RUN apt-get install -y ffmpeg

RUN apt-get update && apt-get install -y rsync sox libsox-dev libsox-fmt-all

RUN curl -L -o /usr/local/bin/mc https://dl.min.io/client/mc/release/linux-amd64/mc &&\
    chmod +x /usr/local/bin/mc

ADD requirements.txt /tmp/requirements.txt
RUN pip install --user -r /tmp/requirements.txt

RUN python -c "import nltk; nltk.download('punkt')"

RUN rm -rf /tmp/*

CMD "/bin/bash"
