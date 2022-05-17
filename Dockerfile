FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /entropy-reproduction

RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    python3-pip

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

CMD ["/bin/bash"]
