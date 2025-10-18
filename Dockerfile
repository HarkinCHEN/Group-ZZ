# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# copy the requirements file first to leverage Docker cache
COPY requirements.txt requirements.txt

# install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#copy the rest of the application code
COPY app.py .

COPY models/model_v02.joblib ./models/

ENV BASE_DIR /app
 
# expose the port the app runs on
EXPOSE 9696

# run the application
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "app:app"]