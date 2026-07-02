# Use the official Python 3.11.15-slim image as the base image
FROM python:3.11.15-slim

# Set the working directory in the container
WORKDIR /app

# Copy your application code into the container
COPY fastapp/ /app/fastapp

COPY models/model_artifact.pkl /app/models/model_artifact.pkl

# Install any required Python packages
RUN pip install --no-cache-dir -r /app/fastapp/requirements.txt

EXPOSE 5000

# Command to run your application when the container starts
#local
CMD ["python", "-m", "fastapp.app"]

#Prod
# CMD ["uvicorn", "fastapp.app:app", "--host", "0.0.0.0", "--port", "5000"]
