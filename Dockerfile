# Use the official Python image.
FROM python:3.9-slim

# Set the working directory.
WORKDIR /app

# Copy the current directory contents into the container.
COPY . /app

# Install any needed packages specified in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860.
EXPOSE 7860

# Run app.py when the container launches.
CMD ["python", "app.py"]
