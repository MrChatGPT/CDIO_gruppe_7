FROM python:3.8-slim
WORKDIR /app  # Set a specific work directory

# Install system packages from a list
COPY packages.txt packages.txt
RUN apt-get update && xargs -a packages.txt apt-get install -y && rm -rf /var/lib/apt/lists/*

# Install python packages from a list
COPY requirements.txt requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt

