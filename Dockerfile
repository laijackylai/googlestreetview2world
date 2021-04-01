FROM continuumio/miniconda3

# RUN mkdir scripts
# VOLUME [ ".:/scripts" ]

WORKDIR /scripts

RUN apt-get update
RUN apt install cmake g++ wget build-essential unzip -y

RUN mkdir raw_img
RUN mkdir depth_maps

# RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
# RUN unzip opencv.zip

# RUN mkdir -p build 
# WORKDIR /scripts/build
# RUN cmake ../opencv-master
# RUN cmake --build .

COPY . .
RUN pip install -U -r requirements.txt

CMD tail -f /dev/null