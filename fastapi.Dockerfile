FROM python:3.9-slim-buster

# Set environment variables
ENV PORT=8000

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY fastapi-requirements.txt .
RUN apt-get update && apt-get install -y gcc python3-dev
RUN pip install --no-cache-dir -r fastapi-requirements.txt

# Expose the app port
EXPOSE $PORT

WORKDIR /app/src
# Define the startup command
CMD ["uvicorn", "fastapiapp:app", "--host", "0.0.0.0", "--port", "8000"]