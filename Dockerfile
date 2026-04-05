# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy the current directory contents into the container at /app
COPY . .

# Railway sets PORT dynamically, default to 5000 for local
ENV PORT=5000
EXPOSE $PORT

# Use gunicorn for production (Railway compatible)
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app.app:app
