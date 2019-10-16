FROM pytorch/pytorch

WORKDIR /work
COPY requirements /work
RUN pip install -r requirements.txt
