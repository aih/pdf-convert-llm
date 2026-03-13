import os
import time
from pypdf import PdfReader, PdfWriter
from google import genai
import re

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY environment variable not set.")
    exit(1)

client = genai.Client(
    api_key=api_key,
    http_options={'timeout': 600000}
)

prompt = """
You are an expert in legislative drafting documents. 
Convert the following PDF content into structured XHTML. 
Use standard HTML tags:
- <h1> for the main chapter title
- <h2>, <h3>, <h4> for section and subsection headings
- <p> for regular text paragraphs
- <ul> and <li> for bulleted lists
- <ol> and <li> for numbered lists
- <strong> or <em> for emphasis where apparent in the text

Do NOT include any XML declarations or markdown formatting backticks (```html, ```xml). 
Return ONLY the raw HTML elements that can be placed directly inside a <body> tag.
"""

pdf_dir = "chapter_pdfs"
out_dir = "xml_content"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])

def convert_pdf_part(pdf_path, part_name):
    print(f"Uploading {part_name} to Gemini...")
    for attempt in range(3):
        try:
            gfile = client.files.upload(file=pdf_path)
            print(f"Generating content for {part_name}...")
            response = client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=[prompt, gfile]
            )
            html_content = response.text
            if html_content.startswith("```html"):
                html_content = html_content[7:]
            if html_content.endswith("```"):
                html_content = html_content[:-3]
            client.files.delete(name=gfile.name)
            return html_content.strip()
        except Exception as e:
            print(f"Error processing {part_name} on attempt {attempt+1}: {e}")
            time.sleep(10)
    return None

def extract_inner_xhtml(xhtml_content):
    # Removes any wrapping <body>, <document>, or chunks and leaves just the inner tags
    content = xhtml_content
    # Quick strip of document/body tags if LLM accidentally added them despite instructions
    content = re.sub(r'^\s*<body>(.*?)</body>\s*$', r'\1', content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r'^\s*<document>(.*?)</document>\s*$', r'\1', content, flags=re.IGNORECASE | re.DOTALL)
    return content.strip()

for filename in pdf_files:
    pdf_path = os.path.join(pdf_dir, filename)
    out_filename = filename.replace('.pdf', '.xhtml')
    out_file_path = os.path.join(out_dir, out_filename)
    
    if os.path.exists(out_file_path) and os.path.getsize(out_file_path) > 100:
        with open(out_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if "Quota exceeded" not in content and "API error" not in content and "chunk_error" not in content:
            print(f"Skipped {filename}, already properly converted.")
            continue

    print(f"Processing {filename} via splitting...")
    
    # Split the PDF
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    if total_pages <= 1:
        # Too small to split sensibly, try it whole
        print("PDF has only 1 page, processing whole...")
        result = convert_pdf_part(pdf_path, filename)
        if result:
            with open(out_file_path, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Saved {out_filename}")
        time.sleep(35)
        continue
        
    midpoint = total_pages // 2
    
    p1_path = os.path.join(out_dir, "temp_part1.pdf")
    p2_path = os.path.join(out_dir, "temp_part2.pdf")
    
    writer1 = PdfWriter()
    for i in range(0, midpoint):
        writer1.add_page(reader.pages[i])
    with open(p1_path, "wb") as fOut:
        writer1.write(fOut)
        
    writer2 = PdfWriter()
    for i in range(midpoint, total_pages):
        writer2.add_page(reader.pages[i])
    with open(p2_path, "wb") as fOut:
        writer2.write(fOut)
        
    print(f"Split {filename} into Part 1 (pages 1-{midpoint}) and Part 2 (pages {midpoint+1}-{total_pages})")
    
    part1_xhtml = convert_pdf_part(p1_path, f"{filename}_part1")
    if not part1_xhtml:
        print(f"Failed to generate part 1 for {filename}")
        continue
        
    print("Waiting 35 seconds to avoid rate limits between parts...")
    time.sleep(35)
    
    part2_xhtml = convert_pdf_part(p2_path, f"{filename}_part2")
    if not part2_xhtml:
        print(f"Failed to generate part 2 for {filename}")
        continue
        
    print(f"Combining XHTML for {filename}...")
    combined = extract_inner_xhtml(part1_xhtml) + "\n\n<!-- PAGE SPLIT -->\n\n" + extract_inner_xhtml(part2_xhtml)
    
    with open(out_file_path, 'w', encoding='utf-8') as f:
        f.write(combined)
        
    print(f"Saved recombined {out_filename}")
    
    # Cleanup temp files
    os.remove(p1_path)
    os.remove(p2_path)
    
    print("Waiting 35 seconds to avoid rate limiting before next file...")
    time.sleep(35)

print("Finished processing all PDFs via splitting.")
