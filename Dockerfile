# Use a Python 3.11 base image
FROM python:3.11-slim

# Define metadata
LABEL maintainer="Padam Singh <padamsingh@iisc.ac.in>"

# Set working directory
WORKDIR /my_app

# Copy required files into working directory
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train the model during the build
RUN python train.py

# Expose port 80
EXPOSE 80

# Run test.py to perform model evaluations
CMD ["python", "test.py"]