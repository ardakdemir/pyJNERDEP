FROM pytorch/pytorch

WORKDIR /work
COPY requirements.txt /work
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install vim -y &&
 apt-get install nmap -y
