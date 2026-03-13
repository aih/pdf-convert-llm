import os
import argparse
import time
import requests
import json
import re

def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to XHTML and configure the Web UI.")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files to convert")
    parser.add_argument("--web-dir", default="web-ui", help="Path to the Web UI directory (default: 'web-ui')")
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL of the running pdf-convert-llm backend (default: http://localhost:8000)")
    parser.add_argument("--llm-provider", default="gemini", choices=["gemini", "anthropic"], help="LLM provider to use")
    parser.add_argument("--model-name", default="gemini-2.5-pro", help="Model name to use")
    parser.add_argument("--title", default="Document Viewer", help="Main title for the website")
    parser.add_argument("--subtitle", default="", help="Subtitle for the website")
    parser.add_argument("--about-text", default="", help="HTML text for the 'About this site' tooltip")
    parser.add_argument("--hide-banner", action="store_true", help="Hide the Ad Hoc prototype banner")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip PDF conversion and just generate config/chapters.json based on existing files in web-ui/public")
    
    args = parser.parse_args()
    
    public_dir = os.path.join(args.web_dir, "public")
    if not os.path.exists(public_dir):
        os.makedirs(public_dir)
        
    config = {
        "title": args.title,
        "headerTitleMain": args.title,
        "headerTitleSub": args.subtitle,
        "showAdHocBanner": not args.hide_banner,
        "aboutHtml": args.about_text
    }
    
    with open(os.path.join(public_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"Config saved to {os.path.join(public_dir, 'config.json')}")
    
    chapters = []
    
    if args.skip_conversion:
        print("Skipping conversion. Building chapters.json from existing .xhtml files...")
        existing_files = sorted([f for f in os.listdir(public_dir) if f.endswith('.xhtml')])
        for out_filename in existing_files:
            file_path = os.path.join(public_dir, out_filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            title_match = re.search(r'<h1.*?>(.*?)</h1>', content, re.IGNORECASE | re.DOTALL)
            if title_match:
                chapter_title = title_match.group(1).strip()
                chapter_title = re.sub(r'<[^>]+>', '', chapter_title)
            else:
                chapter_title = out_filename.replace('.xhtml', '')
            chapters.append({
                "number": "",
                "title": chapter_title,
                "filename": out_filename
            })
        
        with open(os.path.join(public_dir, "chapters.json"), "w") as f:
            json.dump(chapters, f, indent=2)
        print(f"Chapters configuration saved to {os.path.join(public_dir, 'chapters.json')}")
        return
        
    prompt = """
You are an expert in document processing. 
Convert the following PDF content into structured XHTML. 
Use standard HTML tags like <h1> for the main chapter title, <h2>, <h3>, <p>, <ul>, <li>, <strong>, etc.
Do NOT include any XML declarations or markdown formatting backticks.
Return ONLY the raw HTML elements that can be placed directly inside a <body> tag.
"""
    
    pdf_files = sorted([f for f in os.listdir(args.pdf_dir) if f.endswith('.pdf')])
    
    for filename in pdf_files:
        pdf_path = os.path.join(args.pdf_dir, filename)
        print(f"Submitting {filename} to API...")
        
        with open(pdf_path, 'rb') as f:
            files = {'files': (filename, f, 'application/pdf')}
            data = {
                'prompt': prompt,
                'llm_provider': args.llm_provider,
                'model_name': args.model_name,
                'processing_mode': 'batch'
            }
            
            try:
                response = requests.post(f"{args.api_url}/convert", files=files, data=data)
            except requests.exceptions.ConnectionError:
                print(f"Error: Could not connect to API at {args.api_url}. Is the backend running?")
                return
                
            if response.status_code != 200:
                print(f"Error submitting {filename}: {response.text}")
                continue
                
            job_id = response.json().get('job_id')
            print(f"Job started. ID: {job_id}")
            
            # Poll for completion
            while True:
                status_response = requests.get(f"{args.api_url}/status/{job_id}")
                if status_response.status_code != 200:
                    print(f"Error checking status for {job_id}")
                    break
                    
                status_data = status_response.json()
                status = status_data.get('status')
                
                if status in ['completed', 'completed_with_errors', 'failed']:
                    print(f"Job {status}: {status_data.get('message')}")
                    break
                    
                print(f"Progress: {status_data.get('progress')}% - {status_data.get('message')}")
                time.sleep(5)
                
            if status in ['completed', 'completed_with_errors']:
                download_response = requests.get(f"{args.api_url}/download/{job_id}")
                if download_response.status_code == 200:
                    results = download_response.json().get('results', [])
                    for result in results:
                        xml_filename = result.get('filename')
                        xml_content = result.get('xml_content')
                        
                        if xml_filename and xml_content:
                            out_filename = xml_filename.replace('.xml', '.xhtml').replace('.pdf', '.xhtml')
                            out_file_path = os.path.join(public_dir, out_filename)
                            
                            with open(out_file_path, 'w', encoding='utf-8') as out_f:
                                out_f.write(xml_content)
                            print(f"Saved {out_filename}")
                            
                            # Try to extract an <h1> title
                            title_match = re.search(r'<h1.*?>(.*?)</h1>', xml_content, re.IGNORECASE | re.DOTALL)
                            if title_match:
                                chapter_title = title_match.group(1).strip()
                                # remove internal tags
                                chapter_title = re.sub(r'<[^>]+>', '', chapter_title)
                            else:
                                chapter_title = out_filename.replace('.xhtml', '')
                                
                            chapters.append({
                                "number": "",
                                "title": chapter_title,
                                "filename": out_filename
                            })
                else:
                    print(f"Failed to download results for {job_id}")
                    
            requests.delete(f"{args.api_url}/jobs/{job_id}")
            print("-" * 40)
            
    with open(os.path.join(public_dir, "chapters.json"), "w") as f:
        json.dump(chapters, f, indent=2)
    print(f"Chapters configuration saved to {os.path.join(public_dir, 'chapters.json')}")
    print("All processing finished.")

if __name__ == "__main__":
    main()
