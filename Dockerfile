FROM python:3.9

WORKDIR /app
ADD . .

RUN apt-get update && apt-get install espeak-ng espeak-ng-espeak -y
RUN pip install -r requirements.txt
RUN mkdir /output

ENV AUDIO_ENABLED=false

EXPOSE 8124

CMD ["python", "./engine.py"] 

