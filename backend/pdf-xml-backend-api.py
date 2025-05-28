import os
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse # Added FileResponse
from fastapi.staticfiles import StaticFiles # Added StaticFiles
from pydantic import BaseModel
import PyPDF2
from anthropic import AsyncAnthropic # Keep for potential future use
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import logging
import pathlib # Added pathlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF to XML Converter API", version="1.0.0")

# CORS middleware (still useful for development or if API is accessed from other origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Configure this properly in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_CHUNK_PAGES = 20
TEMP_DIR_ROOT = Path(tempfile.gettempdir()) / "pdf_converter_jobs"
TEMP_DIR_ROOT.mkdir(exist_ok=True) # Ensure root temp dir exists

# Initialize LLM clients
anthropic_client = AsyncAnthropic(api_key=CLAUDE_API_KEY) if CLAUDE_API_KEY else None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GOOGLE_API_KEY not found. Gemini functionality will be unavailable.")

if not CLAUDE_API_KEY:
    logger.warning("ANTHROPIC_API_KEY not found. Claude functionality will be unavailable.")


# Data models
class ProcessingRequest(BaseModel):
    prompt: str
    llm_provider: str = "gemini"
    model_name: Optional[str] = None

class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int  # 0-100
    current_chunk: int
    total_chunks: int
    message: str
    results: Optional[List[Dict[str, Any]]] = None

# In-memory job storage (use Redis or similar for production)
jobs: Dict[str, ProcessingStatus] = {}

class PDFProcessor:
    def __init__(self):
        self.anthropic = anthropic_client

    async def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        chunks = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                if total_pages == 0:
                    logger.warning(f"PDF {pdf_path.name} has 0 pages or is unreadable.")
                    # Return a single chunk indicating an issue, or raise an error
                    return [{
                        "chunk_index": 0, "start_page": 0, "end_page": 0,
                        "text": "Error: PDF has no pages or is unreadable.", "page_count": 0,
                        "error": "PDF has no pages or is unreadable."
                    }]


                for i in range(0, total_pages, MAX_CHUNK_PAGES):
                    start_page = i
                    end_page = min(i + MAX_CHUNK_PAGES, total_pages)
                    chunk_text = ""
                    for page_num in range(start_page, end_page):
                        try:
                            page = pdf_reader.pages[page_num]
                            extracted_page_text = page.extract_text()
                            if extracted_page_text:
                                chunk_text += f"\n--- Page {page_num + 1} ---\n"
                                chunk_text += extracted_page_text
                            else:
                                chunk_text += f"\n--- Page {page_num + 1} (no text extracted) ---\n"
                        except Exception as e:
                            logger.warning(f"Error extracting page {page_num + 1} from {pdf_path.name}: {str(e)}")
                            chunk_text += f"\n--- Page {page_num + 1} (extraction error: {str(e)}) ---\n"

                    chunks.append({
                        "chunk_index": len(chunks),
                        "start_page": start_page + 1,
                        "end_page": end_page,
                        "text": chunk_text.strip(),
                        "page_count": end_page - start_page
                    })
            return chunks
        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"PyPDF2 error reading PDF {pdf_path.name}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error reading PDF {pdf_path.name}: Invalid or corrupted PDF file. {str(e)}")
        except Exception as e:
            logger.error(f"General error extracting text from PDF {pdf_path.name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing PDF {pdf_path.name}: {str(e)}")

    async def process_chunk_with_claude(self, chunk_text: str, prompt: str, model_name: str = "claude-3-haiku-20240307") -> str:
        if not self.anthropic:
            raise HTTPException(status_code=500, detail="Claude API key not configured or client not initialized.")
        try:
            full_prompt = f"{prompt}\n\nPDF Content Chunk:\n{chunk_text}\n\nReturn ONLY the XML content for this chunk, suitable for insertion within a larger document's <content> tag."
            message = await self.anthropic.messages.create(
                model=model_name,
                max_tokens=4000, # Consider making this configurable
                temperature=0.1,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Claude API error: {str(e)}")

    async def process_chunk_with_gemini(self, chunk_text: str, prompt: str, model_name: str = "gemini-1.5-flash") -> str:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="Gemini API key not configured.")
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings={ # Adjust as needed, more restrictive might be safer
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
                generation_config={"response_mime_type": "text/plain"} # Explicitly ask for plain text
            )
            full_prompt = f"{prompt}\n\nPDF Content Chunk:\n{chunk_text}\n\nReturn ONLY the XML content for this chunk, suitable for insertion within a larger document's <content> tag."
            response = await model.generate_content_async(full_prompt)

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 raise Exception(f"Gemini content generation blocked: {response.prompt_feedback.block_reason.name}. Adjust safety settings or prompt.")

            if not response.text: # Check if response.text is empty or None
                # Investigate parts if text is missing
                logger.warning(f"Gemini response.text is empty. Parts: {response.parts}")
                if response.parts:
                    # Try to construct text from parts if available
                    return "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
                raise Exception("Empty response from Gemini and no usable parts.")

            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            # Check for specific Gemini API errors if possible
            raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    async def process_chunk(self, chunk: Dict[str, Any], prompt: str, llm_provider: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        if "error" in chunk: # If PDF extraction already failed for this "chunk"
             return {**chunk, "xml_content": f"<!-- Error during PDF text extraction: {chunk['error']} -->", "status": "error"}
        try:
            if llm_provider.lower() == "claude":
                model = model_name or "claude-3-haiku-20240307" # Default Claude model
                xml_content = await self.process_chunk_with_claude(chunk["text"], prompt, model)
            elif llm_provider.lower() == "gemini":
                model = model_name or "gemini-1.5-flash" # Default Gemini model (ensure this is a valid name)
                xml_content = await self.process_chunk_with_gemini(chunk["text"], prompt, model)
            else:
                raise HTTPException(status_code=400, detail="Unsupported LLM provider specified.")

            return {
                "chunk_index": chunk["chunk_index"], "start_page": chunk["start_page"],
                "end_page": chunk["end_page"], "xml_content": xml_content, "status": "completed"
            }
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.get('chunk_index', 'N/A')}: {str(e)}")
            return {
                "chunk_index": chunk.get("chunk_index"), "start_page": chunk.get("start_page"),
                "end_page": chunk.get("end_page"), "xml_content": f"<!-- Error processing chunk with LLM: {str(e)} -->",
                "status": "error", "error": str(e)
            }

    def clean_xml_content(self, xml_string: str) -> str:
        xml_string = re.sub(r'```xml\s*', '', xml_string, flags=re.IGNORECASE)
        xml_string = re.sub(r'```\s*$', '', xml_string)
        xml_string = xml_string.strip()
        if not xml_string:
            return "<!-- LLM returned empty content for this chunk -->"
        try:
            # Attempt to parse to ensure it's well-formed, then pretty print
            root = ET.fromstring(xml_string) # This validates basic well-formedness
            # Pretty print
            rough_string = ET.tostring(root, 'unicode', method='xml')
            reparsed = minidom.parseString(rough_string)
            # Get pretty XML, remove XML declaration from individual chunks
            pretty_xml = reparsed.toprettyxml(indent="  ")
            pretty_xml = re.sub(r'^<\?xml version="1.0" \?>\n?', '', pretty_xml, count=1).strip()
            return pretty_xml
        except ET.ParseError as e:
            logger.warning(f"Failed to parse LLM XML output: {e}. Returning as unparsed text within a comment/wrapper.")
            # Escape the problematic string to be safely included as text or in a comment
            escaped_xml_string = xml_string.replace('&', '&').replace('<', '<').replace('>', '>')
            return f"<unparsed_chunk_error comment=\"LLM output was not well-formed XML. ParseError: {str(e)}\">\n  <original_text>{escaped_xml_string}</original_text>\n</unparsed_chunk_error>"


    def stitch_xml_results(self, results: List[Dict[str, Any]], original_filename: str) -> str:
        try:
            results.sort(key=lambda x: x.get("chunk_index", 0))
            total_pages = 0
            if results: # Calculate total_pages from the last chunk if results exist
                last_chunk_with_page = next((res for res in reversed(results) if "end_page" in res and isinstance(res["end_page"], int)), None)
                if last_chunk_with_page:
                    total_pages = last_chunk_with_page["end_page"]

            processing_date = datetime.now().isoformat()
            filename_no_ext = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename

            # Build the XML structure using ElementTree for better control
            doc_root = ET.Element("document")
            metadata_el = ET.SubElement(doc_root, "metadata")
            ET.SubElement(metadata_el, "title").text = filename_no_ext
            ET.SubElement(metadata_el, "source_filename").text = original_filename
            ET.SubElement(metadata_el, "page_count").text = str(total_pages)
            ET.SubElement(metadata_el, "processing_date").text = processing_date
            ET.SubElement(metadata_el, "chunks_processed").text = str(len(results))

            content_el = ET.SubElement(doc_root, "content")

            for result in results:
                chunk_comment = f"Pages {result.get('start_page', 'N/A')}-{result.get('end_page', 'N/A')}"
                if result.get("status") == "completed" and result.get("xml_content"):
                    cleaned_xml_chunk = self.clean_xml_content(result["xml_content"])
                    try:
                        # Try to parse the cleaned chunk and append its children
                        # This assumes clean_xml_content returns a string representing one or more sibling XML elements
                        # Wrap in a temporary root to parse if it's multiple elements or just text
                        chunk_wrapper_str = f"<temp_wrapper>{cleaned_xml_chunk}</temp_wrapper>"
                        chunk_root = ET.fromstring(chunk_wrapper_str)
                        content_el.append(ET.Comment(f" Start of chunk: {chunk_comment} "))
                        for child_el in chunk_root:
                            content_el.append(child_el)
                        content_el.append(ET.Comment(f" End of chunk: {chunk_comment} "))
                    except ET.ParseError as e:
                        logger.error(f"Error parsing cleaned XML chunk for {chunk_comment}: {e}. Original content: {result['xml_content'][:200]}")
                        content_el.append(ET.Comment(f" Error processing chunk {chunk_comment}: Could not parse LLM output. {str(e)} "))
                        # Add raw content as escaped text or within an error element
                        error_chunk_el = ET.SubElement(content_el, "chunk_processing_error", attrib={"comment": str(e)})
                        error_chunk_el.text = result["xml_content"] # or cleaned_xml_chunk
                else:
                    error_msg = result.get('error', 'Unknown error or empty content')
                    content_el.append(ET.Comment(f" Error in chunk {chunk_comment}: {error_msg} "))
                    error_el = ET.SubElement(content_el, "chunk_error", attrib={"details": error_msg})
                    if "xml_content" in result and result["xml_content"]: # Include the problematic content if any
                         error_el.text = result["xml_content"]


            # Pretty print the final XML document
            rough_string = ET.tostring(doc_root, encoding='unicode', method='xml')
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ")

        except Exception as e:
            logger.error(f"Critical error stitching XML results for {original_filename}: {str(e)}")
            # Fallback to a simple error XML
            return f"<document><error>Error stitching XML chunks for {original_filename}: {str(e)}</error></document>"

processor = PDFProcessor()

# --- API Endpoints ---

async def process_single_file_task(
    job_id: str,
    file: UploadFile,
    prompt: str,
    llm_provider: str,
    model_name: Optional[str],
    job_temp_dir: Path
):
    jobs[job_id].status = "processing"
    jobs[job_id].message = f"Preparing {file.filename}..."
    
    temp_file_path = job_temp_dir / file.filename
    try:
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        await file.close() # Ensure file is closed

        jobs[job_id].message = f"Extracting text from {file.filename}..."
        chunks = await processor.extract_text_from_pdf(temp_file_path)
        
        if not chunks or (len(chunks) == 1 and "error" in chunks[0]):
            error_message = chunks[0]["error"] if chunks and "error" in chunks[0] else "No processable content found in PDF."
            logger.error(f"Failed to extract any processable chunks from {file.filename}: {error_message}")
            jobs[job_id].status = "failed"
            jobs[job_id].message = f"Failed: {file.filename} - {error_message}"
            jobs[job_id].results = [{
                "filename": file.filename.replace('.pdf', '.xml'), "original_filename": file.filename,
                "xml_content": f"<!-- Error: {error_message} -->", "chunks_processed": 0, "status": "failed"
            }]
            return

        jobs[job_id].total_chunks = len(chunks)
        jobs[job_id].current_chunk = 0
        
        chunk_results = []
        for i, chunk_data in enumerate(chunks):
            jobs[job_id].current_chunk = i + 1
            jobs[job_id].progress = int((jobs[job_id].current_chunk / jobs[job_id].total_chunks) * 90) # 90% for processing, 10% for stitching
            jobs[job_id].message = f"LLM processing chunk {jobs[job_id].current_chunk}/{jobs[job_id].total_chunks} of {file.filename}"
            
            processed_chunk = await processor.process_chunk(chunk_data, prompt, llm_provider, model_name)
            chunk_results.append(processed_chunk)
        
        jobs[job_id].message = f"Stitching XML for {file.filename}..."
        xml_content = processor.stitch_xml_results(chunk_results, file.filename)
        
        final_status = "completed" if all(cr.get('status') == 'completed' for cr in chunk_results) else "completed_with_errors"
        if not any(cr.get('status') == 'completed' for cr in chunk_results):
            final_status = "failed"
            xml_content = f"<!-- All chunks failed to process for {file.filename} -->"


        jobs[job_id].results = [{
            "filename": file.filename.replace('.pdf', '.xml'),
            "original_filename": file.filename,
            "xml_content": xml_content,
            "chunks_processed": len(chunk_results),
            "status": final_status
        }]
        jobs[job_id].status = "completed" # Or final_status if you want more granular job status
        jobs[job_id].progress = 100
        jobs[job_id].message = f"Completed: {file.filename}"

    except HTTPException as e: # Catch HTTPExceptions from PDFProcessor
        logger.error(f"HTTP Exception during processing {file.filename} for job {job_id}: {e.detail}")
        jobs[job_id].status = "failed"
        jobs[job_id].message = f"Failed: {file.filename} - {e.detail}"
        jobs[job_id].results = [{"filename": file.filename.replace('.pdf', '.xml'), "original_filename": file.filename, "xml_content": f"<!-- Error: {e.detail} -->", "chunks_processed": 0, "status": "failed"}]
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename} for job {job_id}: {str(e)}", exc_info=True)
        jobs[job_id].status = "failed"
        jobs[job_id].message = f"Failed: {file.filename} - Unexpected error: {str(e)}"
        jobs[job_id].results = [{"filename": file.filename.replace('.pdf', '.xml'), "original_filename": file.filename, "xml_content": f"<!-- Unexpected Error: {str(e)} -->", "chunks_processed": 0, "status": "failed"}]
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()

# --- process_files_background_batch now takes file_contents_cache ---
async def process_files_background_batch(
    job_id: str,
    # files: List[UploadFile], # NO LONGER TAKES UploadFile LIST
    file_contents_cache: List[tuple[str, bytes]], # NOW TAKES THIS
    prompt: str,
    llm_provider: str,
    model_name: Optional[str],
    job_temp_dir: Path
):
    jobs[job_id].status = "processing"
    # The message "Reading uploaded files into memory..." is no longer needed here
    # as it's done before this task starts.
    jobs[job_id].message = "Preparing for batch processing..." # Adjusted message
    
    batch_results = []
    total_files_to_process = len(file_contents_cache) # Use length of cache

    # The initial read loop is GONE from here, as it's done in /convert

    # --- Estimation Loop (uses cached content from argument) ---
    estimated_total_chunks = 0
    temp_file_paths_for_estimation = []
    jobs[job_id].message = "Analyzing files for chunk estimation..."

    for i, (original_filename_from_cache, content_bytes_from_cache) in enumerate(file_contents_cache):
        try:
            _temp_path = job_temp_dir / f"estimate_{original_filename_from_cache}"
            with open(_temp_path, "wb") as tf:
                tf.write(content_bytes_from_cache)
            
            temp_file_paths_for_estimation.append(_temp_path)
            chunks = await processor.extract_text_from_pdf(_temp_path)
            estimated_total_chunks += len(chunks)
        except Exception as e:
            logger.warning(f"Could not estimate chunks for {original_filename_from_cache}: {e}")
            estimated_total_chunks += 1
        
        # Progress for estimation (e.g., first 10% of overall job progress)
        current_progress_estimation = int(((i + 1) / total_files_to_process) * 10) if total_files_to_process > 0 else 0
        jobs[job_id].progress = min(current_progress_estimation, 10)

    for _path in temp_file_paths_for_estimation:
        if _path.exists(): _path.unlink()

    jobs[job_id].total_chunks = estimated_total_chunks
    jobs[job_id].current_chunk = 0

    # --- Main Processing Loop (uses cached content from argument) ---
    for file_idx, (original_filename, file_content_bytes) in enumerate(file_contents_cache):
        # ... (the rest of this loop remains IDENTICAL to the previous version,
        # as it already operates on original_filename and file_content_bytes) ...
        # ... Make sure any progress calculation is based on total_files_to_process ...
        file_specific_job_message_prefix = f"File {file_idx+1}/{total_files_to_process} ({original_filename})"
        jobs[job_id].message = f"{file_specific_job_message_prefix}: Preparing..."

        temp_file_path = job_temp_dir / original_filename
        try:
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content_bytes) # Write from cached memory
            
            jobs[job_id].message = f"{file_specific_job_message_prefix}: Extracting text..."
            chunks = await processor.extract_text_from_pdf(temp_file_path)

            if not chunks or (len(chunks) == 1 and "error" in chunks[0]):
                error_message = chunks[0]["error"] if chunks and "error" in chunks[0] else "No processable content."
                logger.error(f"No processable chunks for {original_filename}: {error_message}")
                batch_results.append({
                    "filename": original_filename.replace('.pdf', '.xml'), "original_filename": original_filename,
                    "xml_content": f"<!-- Error: {error_message} -->", "chunks_processed": 0, "status": "failed"
                })
                jobs[job_id].current_chunk += (len(chunks) if chunks else 1) 
                continue

            chunk_results_for_file = []
            for chunk_data in chunks:
                jobs[job_id].current_chunk += 1
                # Base overall progress from 10% (estimation) up to 90% (processing)
                overall_progress = 10 + int((jobs[job_id].current_chunk / jobs[job_id].total_chunks) * 80) if jobs[job_id].total_chunks > 0 else 10
                jobs[job_id].progress = min(overall_progress, 90)
                
                jobs[job_id].message = f"{file_specific_job_message_prefix}: LLM processing chunk {chunk_data.get('chunk_index',0)+1}/{len(chunks)}"
                processed_chunk = await processor.process_chunk(chunk_data, prompt, llm_provider, model_name)
                chunk_results_for_file.append(processed_chunk)
            
            jobs[job_id].message = f"{file_specific_job_message_prefix}: Stitching XML..."
            xml_content = processor.stitch_xml_results(chunk_results_for_file, original_filename)
            
            file_status = "completed"
            if not any(cr.get('status') == 'completed' for cr in chunk_results_for_file):
                file_status = "failed"
            elif any(cr.get('status') != 'completed' for cr in chunk_results_for_file):
                 file_status = "completed_with_errors"

            batch_results.append({
                "filename": original_filename.replace('.pdf', '.xml'),
                "original_filename": original_filename,
                "xml_content": xml_content,
                "chunks_processed": len(chunk_results_for_file),
                "status": file_status
            })

        except Exception as e:
            logger.error(f"Error processing {original_filename} in main loop (job {job_id}): {str(e)}", exc_info=True)
            batch_results.append({
                "filename": original_filename.replace('.pdf', '.xml'),
                "original_filename": original_filename,
                "xml_content": f"<!-- Unexpected Error during processing: {str(e)} -->",
                "chunks_processed": 0, 
                "status": "failed"
            })
            jobs[job_id].current_chunk += (len(chunks) if 'chunks' in locals() and chunks else 1) 
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()
    
    jobs[job_id].results = batch_results
    jobs[job_id].status = "completed"
    jobs[job_id].progress = 100 # Final 10% for stitching and finalization
    jobs[job_id].message = f"Batch processing completed. {len(batch_results)} files processed."


@app.post("/convert")
async def convert_pdf(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    prompt: str = Form(...),
    llm_provider: str = Form(default="gemini"),
    model_name: Optional[str] = Form(default=None),
    processing_mode: str = Form(default="batch"),
):
    # ... (API key checks, file type validation) ...
    if llm_provider.lower() == "claude" and not CLAUDE_API_KEY:
        raise HTTPException(status_code=500, detail="Claude API key not configured. Cannot use Claude.")
    if llm_provider.lower() == "gemini" and not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured. Cannot use Gemini.")
    if llm_provider.lower() not in ["claude", "gemini"]:
        raise HTTPException(status_code=400, detail="Invalid LLM provider. Must be 'claude' or 'gemini'.")

    for file_check in files: # Renamed variable to avoid conflict
        if not file_check.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file_check.filename} is not a PDF.")

    job_id = str(uuid.uuid4())
    job_temp_dir = TEMP_DIR_ROOT / job_id
    job_temp_dir.mkdir(exist_ok=True)

    # --- Read file contents HERE, before starting background task ---
    file_contents_cache: List[tuple[str, bytes]] = []
    logger.info(f"Job {job_id}: Reading {len(files)} files in /convert endpoint.")
    for uploaded_file_obj in files:
        try:
            logger.info(f"Job {job_id}: Reading {uploaded_file_obj.filename} into memory.")
            content_bytes = await uploaded_file_obj.read()
            if content_bytes is None: # Should not happen, read() raises error or returns bytes
                 raise IOError(f"Read returned None for {uploaded_file_obj.filename}")
            logger.info(f"Job {job_id}: Successfully read {len(content_bytes)} bytes for {uploaded_file_obj.filename}.")
            file_contents_cache.append((uploaded_file_obj.filename, content_bytes))
        except Exception as e:
            logger.error(f"Job {job_id}: Failed to read {uploaded_file_obj.filename} in /convert endpoint: {e}", exc_info=True)
            # Clean up job temp dir if created
            if job_temp_dir.exists():
                try:
                    for item in job_temp_dir.iterdir(): item.unlink()
                    job_temp_dir.rmdir()
                except Exception as cleanup_err:
                    logger.error(f"Job {job_id}: Error cleaning up temp dir after read failure: {cleanup_err}")
            raise HTTPException(status_code=500, detail=f"Failed to read uploaded file {uploaded_file_obj.filename}. Error: {str(e)[:100]}")
        finally:
            # It's good practice to ensure the UploadFile is 'finished' with,
            # though FastAPI should handle its lifecycle after request completion.
            # Calling close() might be risky if FastAPI still needs it.
            # Let's rely on FastAPI's cleanup.
            pass # await uploaded_file_obj.close() # Generally not recommended to call manually

    if not file_contents_cache: # Should not happen if 'files' is not empty and no error raised
        raise HTTPException(status_code=400, detail="No files were successfully read.")

    jobs[job_id] = ProcessingStatus(
        job_id=job_id, status="pending", progress=0, current_chunk=0, total_chunks=0,
        message="Job accepted, files read. Starting background processing...", results=[]
    )

    logger.info(f"Job {job_id}: Starting background task with {len(file_contents_cache)} file(s) from cache. Mode: {processing_mode}")
    background_tasks.add_task(
        process_files_background_batch,
        job_id,
        file_contents_cache, # Pass the cache, not the UploadFile list
        prompt,
        llm_provider,
        model_name,
        job_temp_dir
    )
    return {"job_id": job_id, "status": "started", "processing_mode": "batch_per_request"} # or "batch"

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "claude_available": CLAUDE_API_KEY is not None,
        "gemini_available": GEMINI_API_KEY is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/download/{job_id}")
async def download_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job.status not in ["completed", "completed_with_errors"]: # Allow download even if some files failed
        raise HTTPException(status_code=400, detail=f"Job not completed yet. Status: {job.status}")
    return {"results": job.results}

@app.delete("/jobs/{job_id}")
async def cleanup_job(job_id: str):
    if job_id in jobs:
        del jobs[job_id]
        job_temp_dir = TEMP_DIR_ROOT / job_id
        if job_temp_dir.exists():
            try:
                # Basic recursive delete, for more robust use shutil.rmtree
                for item in job_temp_dir.iterdir():
                    if item.is_dir():
                        # If subdirectories are created, this needs to be recursive
                        # For now, assuming only files in job_temp_dir
                        item.unlink() if item.is_file() else os.rmdir(item) # simplistic
                    else:
                        item.unlink()
                job_temp_dir.rmdir()
                logger.info(f"Cleaned up temp directory for job {job_id}")
            except Exception as e:
                logger.error(f"Error cleaning up temp directory for job {job_id}: {e}")
        return {"message": "Job data and temporary files cleaned up successfully"}
    else:
        raise HTTPException(status_code=404, detail="Job not found for cleanup")

# --- Static File Serving (for combined frontend/backend) ---
STATIC_FILES_DIR = pathlib.Path(__file__).resolve().parent / "static_frontend"

# Serve static assets (JS, CSS, images) from React's build output (e.g., /static/*)
if (STATIC_FILES_DIR / "assets").exists(): # For Vite, usually 'assets'
    app.mount("/assets", StaticFiles(directory=(STATIC_FILES_DIR / "assets")), name="vite_assets")
    logger.info(f"Mounted Vite assets from: {STATIC_FILES_DIR / 'assets'}")
elif (STATIC_FILES_DIR / "static").exists(): # For CRA, usually 'static'
    app.mount("/static", StaticFiles(directory=(STATIC_FILES_DIR / "static")), name="static_assets")
    logger.info(f"Mounted CRA static assets from: {STATIC_FILES_DIR / 'static'}")
else:
    logger.warning(f"Static assets sub-directory ('assets' or 'static') not found in {STATIC_FILES_DIR}. Static files might not be served correctly.")

@app.get("/api/info") # New path for API info, to not conflict with root serving index.html
async def api_info():
    return {"message": "PDF to XML Converter API", "version": "1.0.0"}


# Catch-all for serving index.html (SPA behavior) and other root static files
# This MUST be one of the LAST routes.
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_react_app_catch_all(full_path: str):
    file_path = STATIC_FILES_DIR / full_path
    if full_path and file_path.exists() and file_path.is_file():
        # Serve specific files like manifest.json, favicon.ico if they exist at the root of static_frontend
        return FileResponse(file_path)
    
    # Default to serving index.html for SPA routing
    index_html_path = STATIC_FILES_DIR / "index.html"
    if index_html_path.exists():
        return FileResponse(index_html_path)
    
    # If index.html is not found, it's likely a setup issue or API call to a non-existent path.
    # FastAPI will return its own 404 if no other route matches.
    # For clarity, we can raise one, but typically this endpoint is for serving the SPA.
    logger.warning(f"Catch-all route: Could not find {file_path} or {index_html_path}. This might be an unhandled API route or missing SPA file.")
    # Let FastAPI's default 404 handling take over if it's not an SPA asset
    raise HTTPException(status_code=404, detail=f"Resource not found: {full_path}")


@app.get("/", include_in_schema=False) # Explicitly serve index.html for the root path
async def serve_root_index():
    index_html_path = STATIC_FILES_DIR / "index.html"
    if index_html_path.exists():
        return FileResponse(index_html_path)
    logger.error(f"CRITICAL: index.html not found at {index_html_path}. SPA cannot be served.")
    raise HTTPException(status_code=404, detail="Application entry point (index.html) not found.")


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Serving static files from: {STATIC_FILES_DIR}")
    if not STATIC_FILES_DIR.exists() or not (STATIC_FILES_DIR / "index.html").exists():
        logger.warning(
            "Static frontend directory ('static_frontend') or 'index.html' not found. "
            "The frontend app might not be served correctly. "
            "Ensure you have built the frontend and copied it to 'static_frontend' next to this script."
        )
    
    if not CLAUDE_API_KEY and not GEMINI_API_KEY:
        logger.error("CRITICAL: NEITHER CLAUDE_API_KEY NOR GOOGLE_API_KEY is set. LLM functionality WILL FAIL.")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)