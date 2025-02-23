# Use the PyTorch image with CUDA
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Install espeak-ng and cleanup
RUN apt-get update && \
    apt-get install -y espeak-ng && \
    rm -rf /var/lib/apt/lists/*

# Create and switch to /app
WORKDIR /app

# Copy your entire local directory into /app in the container
COPY . /app

# Install poetry
RUN pip install --upgrade pip && \
    pip install poetry

# Install dependencies using poetry
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --with compile

# Expose port 7861 so Docker is aware of it
EXPOSE 7861

# By default, run your FastAPI script
CMD ["python3", "zonos_api.py"]
