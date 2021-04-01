FROM continuumio/miniconda3

# RUN mkdir scripts
# VOLUME [ ".:/scripts" ]

WORKDIR /scripts

RUN apt-get update
RUN apt install g++ -y
RUN apt install build-essential -y

COPY . .
RUN pip install -U -r requirements.txt

CMD tail -f /dev/null