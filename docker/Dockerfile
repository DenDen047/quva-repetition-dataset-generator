FROM python:3.8.5

RUN apt-get update -y
RUN apt-get install -y libgl1-mesa-glx

# for youtube-dl
RUN apt-get install -y ffmpeg

# pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# clean
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*