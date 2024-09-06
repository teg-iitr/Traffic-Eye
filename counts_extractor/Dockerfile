FROM python:3.9-slim
#WORKDIR /usr/src/app
WORKDIR /app
COPY requirements.txt ./
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN (apt-get autoremove -y; \
    apt-get autoclean -y)
CMD ["bash"]