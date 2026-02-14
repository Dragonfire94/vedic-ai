FROM python:3.11-slim

WORKDIR /app

# Install build tools for pyswisseph
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    fonts-nanum \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY backend/requirements.txt ./requirements.txt

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend ./backend

# Ensure fonts are discoverable
RUN fc-cache -fv

# Expose port
EXPOSE 8000

# Run using module path to avoid import errors
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
