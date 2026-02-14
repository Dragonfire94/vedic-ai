FROM python:3.11-slim

WORKDIR /app

# Install build tools for pyswisseph
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    ca-certificates \
    fonts-nanum \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY backend/requirements.txt ./requirements.txt

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Swiss Ephemeris data files (.se*) for high-precision mode.
# Provide an archive URL at build time:
#   --build-arg SWE_EPHE_URL=https://.../se_ephe.tar.xz
# The archive is expected to contain ephemeris files at its root.
ARG SWE_EPHE_URL=""
ARG SWE_ENFORCE_EPHE=1
RUN mkdir -p /usr/share/libswe/ephe \
    && if [ -n "$SWE_EPHE_URL" ]; then \
        curl -fsSL "$SWE_EPHE_URL" -o /tmp/se_ephe.tar.xz \
        && tar -xJf /tmp/se_ephe.tar.xz -C /usr/share/libswe/ephe \
        && rm -f /tmp/se_ephe.tar.xz; \
       fi \
    && if [ "$SWE_ENFORCE_EPHE" = "1" ] && [ -z "$(ls -A /usr/share/libswe/ephe 2>/dev/null)" ]; then \
        echo "Missing Swiss Ephemeris files under /usr/share/libswe/ephe"; \
        echo "Set SWE_EPHE_URL at build time, or set SWE_ENFORCE_EPHE=0 for non-production."; \
        exit 1; \
       fi

ENV SWE_EPHE_PATH=/usr/share/libswe/ephe
ENV SWE_REQUIRE_SWIEPH=1

# Copy backend source
COPY backend ./backend
COPY assets ./assets

# Ensure fonts are discoverable
RUN fc-cache -fv

# Expose port
EXPOSE 8000

# Run using module path to avoid import errors
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
