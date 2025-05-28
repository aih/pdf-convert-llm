# Stage 1: Build React Frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package.json and package-lock.json (or yarn.lock)
COPY frontend/package*.json ./
# COPY frontend/yarn.lock ./ # If using yarn

# Install frontend dependencies
RUN npm install
# RUN yarn install # If using yarn

# Copy the rest of the frontend application code
COPY frontend/ ./

# Build the frontend
RUN npm run build
# RUN yarn build # If using yarn
# Vite outputs to /app/frontend/dist by default

# Stage 2: Build Python Backend and Serve
FROM python:3.10-slim

WORKDIR /app

# Set environment variables (can be overridden by docker-compose)
ENV ANTHROPIC_API_KEY=""
ENV GOOGLE_API_KEY=""
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install OS dependencies if any (e.g., for some PDF libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-os-lib && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend application code
COPY backend/ ./backend/

# Copy the built frontend assets from the frontend-builder stage
# The target directory 'static_frontend' inside 'backend' must match what STATIC_FILES_DIR expects
COPY --from=frontend-builder /app/frontend/dist ./backend/static_frontend/

WORKDIR /app/backend

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# Ensure your pdf-xml-backend-api.py uses host="0.0.0.0"
CMD ["uvicorn", "pdf-xml-backend-api:app", "--host", "0.0.0.0", "--port", "8000"]