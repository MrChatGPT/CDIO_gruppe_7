# simple python container
FROM python:3.9-slim

WORKDIR /app  # Set a specific work directory

# Install system packages from a list
COPY packages.txt packages.txt
RUN apt-get update && \
    apt-get install -y $(cat packages.txt) && \
    rm -rf /var/lib/apt/lists/*

# Install python packages from a list
COPY requirements.txt requirements.txt 
RUN pip3 install --no-cache-dir -r requirements.txt



