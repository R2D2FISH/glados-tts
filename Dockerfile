FROM python:3.9-slim
ADD . .
RUN apt-get update && apt-get install -y curl unzip && \
    curl -L "https://drive.usercontent.google.com/download?id=1TRJtctjETgVVD5p7frSVPmgw8z8FFtjD&confirm=xxx" -o models.zip && \
    unzip models.zip && \
    rm models.zip && \
    apt-get remove -y curl unzip && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r ./requirements.txt
RUN mkdir -p audio
EXPOSE 8124
CMD ["python", "./engine.py"]