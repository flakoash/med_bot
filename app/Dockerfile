FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files and FAISS index
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# start Chainlit app
CMD ["chainlit", "run", "chainlit.py", "--host", "0.0.0.0", "--port", "8000"]
