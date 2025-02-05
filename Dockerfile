# Step 1: Use an official Python runtime as the base image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the current directory content (project files) into the container's working directory
COPY . /app

# Step 4: Install dependencies
# Copy the requirements.txt file first to leverage Docker caching for dependencies
COPY requirements.txt /app/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose port 8000 for FastAPI to listen on
EXPOSE 8000

# Step 6: Run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
