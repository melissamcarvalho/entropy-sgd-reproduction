FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /entropy-reproduction

RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    python3-pip

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN pip install --upgrade torch torchvision

CMD ["/bin/bash"]
