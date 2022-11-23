FROM python:3.9 as builder

WORKDIR /app
ADD . .

RUN apt-get update && apt-get install espeak-ng espeak-ng-espeak portaudio19-dev -y
RUN pip install -r requirements.txt
RUN mkdir /output

ENV AUDIO_ENABLED=false

EXPOSE 5000

CMD ["python", "./glados.py"] 

