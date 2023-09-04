FROM python:3.9-slim-buster

# Set environment variables
ENV PORT=8501

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY streamlit-requirements.txt .
RUN apt-get update && apt-get install -y gcc python3-dev
RUN pip install --no-cache-dir -r streamlit-requirements.txt

# Copy app files
# COPY /src/streamlit.py .
# COPY data/ data/

# Expose the app port
EXPOSE $PORT

WORKDIR /app/src
# Define the startup command
CMD ["streamlit", "run", "streamlit.py"]
# CMD streamlit run --server.port $PORT app.py