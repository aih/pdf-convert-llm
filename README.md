```markdown
# PDF to XML Converter Application

This application allows users to upload PDF files, specify processing instructions (or use a default prompt), and convert the PDFs into structured XML format using a Large Language Model (LLM) via a backend API.

## Features

*   Upload single or multiple PDF files.
*   Upload entire directories of PDF files.
*   Customize XML structure and processing rules via a prompt.
*   Backend processing using Gemini or Claude LLMs (configurable).
*   Files are processed in chunks and results are stitched together.
*   Download individual XML files or all successfully converted XMLs.
*   Real-time progress updates during conversion.

## Project Structure

```
.
├── backend/                  # FastAPI Python backend
│   ├── pdf-xml-backend-api.py  # Main API logic
│   ├── requirements.txt      # Python dependencies
│   └── static_frontend/      # (Will be populated by frontend build)
├── frontend/                 # React (Vite + TypeScript) frontend
│   ├── public/
│   ├── src/                  # Frontend source code
│   │   ├── pdf-to-xml-converter.tsx
│   │   └── global.d.ts       # Custom type declarations
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── Dockerfile                # For building the combined application image
├── docker-compose.yml        # For running the application with Docker Compose
└── README.md                 # This file
```

## Prerequisites

*   **Docker & Docker Compose:** For running the application with Docker (recommended for ease of use). Install from [Docker Desktop](https://www.docker.com/products/docker-desktop/).
*   **Node.js & npm (or yarn):** For building and running the frontend manually (e.g., Node.js v18+).
*   **Python:** For running the backend manually (e.g., Python 3.10+).
*   **API Keys:**
    *   **Google Gemini API Key:** Obtain from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   **(Optional) Anthropic Claude API Key:** Obtain from [Anthropic Console](https://console.anthropic.com/dashboard) if you wish to use Claude.

## Option A: Build and Run with Docker (Recommended)

This method builds the frontend, copies it to the backend, and runs the combined application in a Docker container.

1.  **Clone the Repository (if you haven't):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a `.env` file:**
    In the project root directory (alongside `docker-compose.yml`), create a file named `.env` and add your API keys:
    ```env
    # .env
    ANTHROPIC_API_KEY=your_anthropic_key_if_using_claude
    GOOGLE_API_KEY=your_google_gemini_api_key
    ```
    **Important:** Add `.env` to your `.gitignore` file to avoid committing API keys.

3.  **Build and Run using Docker Compose:**
    Navigate to the project root directory in your terminal and run:
    ```bash
    docker compose up --build
    ```
    *   `--build`: Forces Docker to rebuild the image, picking up any code changes.
    *   To run in detached (background) mode, add the `-d` flag: `docker compose up --build -d`

4.  **Access the Application:**
    Open your web browser and go to `http://localhost:8000`.

5.  **View Logs (if running detached or for debugging):**
    ```bash
    docker compose logs -f pdf-converter-app
    ```
    (Replace `pdf-converter-app` if your service name in `docker-compose.yml` is different).

6.  **Stop the Application:**
    Press `Ctrl+C` in the terminal where `docker compose up` is running. If running detached, use:
    ```bash
    docker compose down
    ```

## Option B: Build and Run Manually

This involves building the frontend separately, then running the backend Python server which will serve both the API and the frontend assets.

### 1. Setup API Keys (Backend)

Set the API keys as environment variables in your terminal session before running the backend:

```bash
export GOOGLE_API_KEY="your_google_gemini_api_key"
# Optional for Claude:
# export ANTHROPIC_API_KEY="your_anthropic_key_if_using_claude"
```

### 2. Build the Frontend

   a. **Navigate to the frontend directory:**
      ```bash
      cd frontend
      ```

   b. **Install dependencies (if first time or `package.json` changed):**
      ```bash
      npm install
      # or
      # yarn install
      ```

   c. **Build the frontend application:**
      ```bash
      npm run build
      # or
      # yarn build
      ```
      This will create a `dist` directory inside `frontend/` containing the static assets.

### 3. Prepare Backend to Serve Frontend

   a. **Navigate to the backend directory:**
      ```bash
      cd ../backend
      # (Assuming you are in frontend/, otherwise adjust path to backend/)
      ```

   b. **Create `static_frontend` directory (if it doesn't exist):**
      ```bash
      mkdir -p static_frontend
      ```

   c. **Copy built frontend assets to the backend:**
      From the `backend` directory, run:
      ```bash
      # Adjust relative path to frontend/dist if your structure is different
      cp -R ../frontend/dist/* ./static_frontend/
      ```
      Ensure that `static_frontend` now contains `index.html`, an `assets` folder, etc.

### 4. Run the Backend API Server

   a. **Navigate to the backend directory (if not already there):**
      ```bash
      cd backend
      ```

   b. **Create a Python virtual environment (recommended):**
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      # On Windows: venv\Scripts\activate
      ```

   c. **Install Python dependencies:**
      ```bash
      pip install -r requirements.txt
      ```

   d. **Run the FastAPI server:**
      ```bash
      python pdf-xml-backend-api.py
      ```
      The server will typically start on `http://0.0.0.0:8000`.

### 5. Access the Application

   Open your web browser and go to `http://localhost:8000`.

### 6. Stop the Backend Server

   Press `Ctrl+C` in the terminal where the Python server is running. If you used a virtual environment, you can deactivate it:
   ```bash
   deactivate
   ```

## Development Notes

*   **Frontend Development Server (for live reload):**
    If you are actively developing the frontend, you can run the Vite development server:
    ```bash
    cd frontend
    npm run dev
    ```
    This usually starts on `http://localhost:5173` (or similar). You would then run the backend API server separately (e.g., on port 8000). In this mode, the frontend will make API calls to `http://localhost:8000`. Ensure your frontend's API fetch calls point to the correct backend URL during development if they are on different ports (often handled by Vite proxy or by temporarily hardcoding during dev). The provided frontend code uses relative paths like `/convert`, which assume same-origin deployment or a proxy.
*   **Backend Live Reload (Uvicorn):**
    For backend development, you can run Uvicorn with live reload:
    ```bash
    cd backend
    # (Activate virtual environment if using one)
    uvicorn pdf-xml-backend-api:app --reload --host 0.0.0.0 --port 8000
    ```

## Configuration

*   **LLM Provider:** The backend defaults to Gemini. This can be changed via the `llm_provider` form data in the frontend request, or by modifying the default in `pdf-xml-backend-api.py`.
*   **Chunk Size:** `MAX_CHUNK_PAGES` in `pdf-xml-backend-api.py` controls the number of pages processed per LLM call.
*   **Default Prompt:** The `defaultPrompt` in `frontend/src/pdf-to-xml-converter.tsx` can be modified.

## Troubleshooting

*   **Docker "command not found":** If `docker compose` fails, ensure Docker Desktop is installed correctly and that the `docker` CLI is in your system's PATH.
*   **Docker "no configuration file provided":** Make sure you are in the project root directory (containing `docker-compose.yml`) when running `docker compose` commands.
*   **API Key Errors:** Double-check that your API keys are correctly set in the `.env` file (for Docker) or as environment variables (for manual runs).
*   **"I/O operation on closed file" (Backend):** This usually indicates an issue with handling uploaded file streams. The current implementation reads files into memory in the `/convert` endpoint before passing to the background task to mitigate this. If it persists, check file sizes and server resources.
*   **Static Files Not Found (404s for CSS/JS):**
    *   Ensure the frontend build (e.g., `frontend/dist/`) was correctly copied to `backend/static_frontend/`.
    *   Verify the `STATIC_FILES_DIR` path and the `app.mount(...)` path in `pdf-xml-backend-api.py` match the structure of your `static_frontend` directory (e.g., `assets` subfolder for Vite, `static` for CRA).
```