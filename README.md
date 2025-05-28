# PDF to XML Converter with FastAPI and React 
This project provides a FastAPI backend for converting PDF files to XML format, along with a React frontend for user interaction. The backend uses a prompt to an LLM + pdf for conversion and serves static files from the frontend build.

## Build Frontend:
In your frontend directory (e.g., pdf-converter-app):
npm run build

This creates a dist directory with the static frontend assets.

## Prepare Backend:
In the Python backend project root (where pdf-xml-backend-api.py is):
Create a directory, e.g., static_frontend.
Copy the contents of your frontend's dist directory into this static_frontend directory.
Example: cp -R ../frontend/dist/* ./static_frontend/ (adjust paths as needed).
static_frontend should now contain index.html, an assets folder (for Vite), etc.

Static file mounting (adjust /assets if Vite outputs a different asset folder name like /static):
if (STATIC_FILES_DIR / "assets").exists(): # For Vite, usually 'assets'
    app.mount("/assets", StaticFiles(directory=(STATIC_FILES_DIR / "assets")), name="vite_assets")
elif (STATIC_FILES_DIR / "static").exists(): # For CRA, usually 'static'
    app.mount("/static", StaticFiles(directory=(STATIC_FILES_DIR / "static")), name="static_assets")
else:
    logger.warning(f"Static assets directory ('assets' or 'static') not found in {STATIC_FILES_DIR}.")
Use code with caution.
Python
Catch-all routes for index.html and other root static files (must be after your API routes):
@app.get("/{full_path:path}", include_in_schema=False)
# ... (serve_react_app_catch_all function) ...

@app.get("/", include_in_schema=False)
# ... (serve_root_index function) ...
Use code with caution.
Python
Ensure your API info endpoint is not at / (e.g., /api/info).
Run Backend:
Set environment variables: ANTHROPIC_API_KEY (if used) and GOOGLE_API_KEY.
From your backend project root:
python pdf-xml-backend-api.py
# or for production-like:
# uvicorn pdf-xml-backend-api:app --host 0.0.0.0 --port 8000

Access the application at http://localhost:8000.
