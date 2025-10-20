# Use an official Python runtime as a parent image
FROM python:3.9-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# copy the requirements file first to leverage Docker cache
COPY requirements.txt requirements.txt

# install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

#copy the rest of the application code
COPY app.py .
COPY train.py .

RUN mkdir -p models && python train.py

ENV BASE_DIR /app
 
# expose the port the app runs on
EXPOSE 9696

# Add health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:9696/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "app:app"]
