import React, { useState, useRef } from 'react';
import { FileText, Download, AlertCircle, CheckCircle, Loader } from 'lucide-react';

const PDFToXMLConverter = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [prompt, setPrompt] = useState('');
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState('');
  const [progressMessage, setProgressMessage] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dirInputRef = useRef<HTMLInputElement>(null);

  const defaultPrompt = `Convert this PDF content chunk to XML.
The content you generate will be part of a larger XML document structured as:
<document>
  <metadata>
    <!-- Populated by the system: e.g., title, author, page_count -->
  </metadata>
  <content>
    <!-- Your generated XML for this chunk will be placed here -->
    <!-- Example for one chunk's content: -->
    <section>
      <heading>Section Title From Chunk</heading>
      <paragraph>Content paragraphs from chunk</paragraph>
      <citation>Reference citations in <ref>...</ref> tags</citation>
      <hyperlink url="...">Link text</hyperlink>
    </section>
    <!-- You can have multiple sections or other elements per chunk -->
  </content>
</document>

For the given PDF content chunk, provide ONLY the XML elements that should go inside the main <content> tag.
Do NOT include the <document>, <metadata>, or the main <content> tags in your response.
Focus on structuring the chunk's text into elements like <section>, <heading>, <paragraph>, etc.
Preserve all text content, maintain document structure within the chunk, and wrap citations in <ref> tags and hyperlinks in <hyperlink> tags with url attributes.
Return only the XML for the chunk's content.`;

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) return;
    const uploadedFiles = Array.from(event.target.files);
    const pdfFiles = uploadedFiles.filter(file => file.type === 'application/pdf');

    if (pdfFiles.length !== uploadedFiles.length) {
      setError('Only PDF files are supported. Please select valid PDF documents.');
      if (event.target) event.target.value = '';
      return;
    }

    setFiles(prevFiles => {
      const newFiles = pdfFiles.filter(pf => !prevFiles.some(ef => ef.name === pf.name && ef.size === pf.size));
      return [...prevFiles, ...newFiles];
    });
    setError('');
    setResults([]);
    if (event.target) event.target.value = '';
  };

  const removeFile = (fileName: string) => {
    setFiles(prevFiles => prevFiles.filter(file => file.name !== fileName));
  };

  const processFile = async (filesToProcess: File[]) => {
    const formData = new FormData();
    filesToProcess.forEach(file => formData.append('files', file));
    formData.append('prompt', prompt || defaultPrompt);
    formData.append('llm_provider', 'gemini');
    formData.append('processing_mode', 'batch');

    try {
      const response = await fetch('/convert', { method: 'POST', body: formData });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error ${response.status}`);
      }
      const data = await response.json();
      return data.job_id;
    } catch (err: any) {
      throw new Error(`Failed to start conversion: ${err.message}`);
    }
  };

  const pollJobStatus = async (jobId: string) => {
    const maxAttempts = 300;
    let attempts = 0;

    const poll = async (): Promise<any[]> => {
      try {
        const response = await fetch(`/status/${jobId}`);
        if (!response.ok) throw new Error(`Status check failed: HTTP ${response.status}`);
        const statusData = await response.json();
        setProgressMessage(statusData.message || `Processing... ${statusData.progress || 0}% complete`);

        if (statusData.status === 'completed') {
          setProgressMessage('Processing complete. Fetching results...');
          const resultsResponse = await fetch(`/download/${jobId}`);
          if (!resultsResponse.ok) throw new Error(`Failed to download results: HTTP ${resultsResponse.status}`);
          const resultsData = await resultsResponse.json();
          return resultsData.results;
        } else if (statusData.status === 'failed') {
          throw new Error(statusData.message || 'Processing failed');
        } else {
          attempts++;
          if (attempts >= maxAttempts) throw new Error('Processing timeout - job took too long.');
          await new Promise(resolve => setTimeout(resolve, 2000));
          return poll();
        }
      } catch (err: any) {
        if (attempts < 3 && (err.message.includes('Status check failed') || err.message.includes('NetworkError'))) {
            attempts++;
            await new Promise(resolve => setTimeout(resolve, 3000 + attempts * 1000));
            return poll();
        }
        throw err;
      }
    };
    return poll();
  };

  const handleProcessFiles = async () => {
    if (files.length === 0) {
      setError('Please upload at least one PDF file to convert.');
      return;
    }
    if (!prompt.trim() && !defaultPrompt.trim()) {
      setError('Processing instructions (prompt) cannot be empty.');
      return;
    }

    setProcessing(true);
    setError('');
    setResults([]);
    setProgressMessage('Initiating conversion process...');

    try {
      const jobId = await processFile(files);
      const processedResults = await pollJobStatus(jobId);
      const transformedResults = processedResults.map((result: any) => ({
        fileName: result.filename,
        content: result.xml_content,
        status: result.status,
        originalFileName: result.original_filename,
        chunksProcessed: result.chunks_processed,
        error_message: result.error_message
      }));
      setResults(transformedResults);
      setFiles([]);

      try {
        await fetch(`/jobs/${jobId}`, { method: 'DELETE' });
      } catch (cleanupError) {
        console.warn('Failed to cleanup job on server:', cleanupError);
      }

    } catch (err: any) {
      console.error('Processing error:', err);
      setError(`Error: ${err.message}`);
      setProgressMessage('');
    } finally {
      setProcessing(false);
    }
  };

  const downloadXML = (result: any) => {
    const blob = new Blob([result.content], { type: 'application/xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = result.fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadAllXML = () => {
    results.filter(r => r.status === 'completed').forEach(downloadXML);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      {/* Hidden file inputs */}
      <input 
        ref={fileInputRef} 
        type="file" 
        multiple 
        accept=".pdf" 
        onChange={handleFileUpload} 
        className="hidden" 
      />
      <input 
        ref={dirInputRef} 
        type="file" 
        {...{directory: "", webkitdirectory: ""}} 
        onChange={handleFileUpload} 
        className="hidden" 
      />

      {/* Main Container - Centered */}
      <div className="container mx-auto max-w-5xl px-4 py-8">
        
        {/* Header */}
        <header className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full mb-6 shadow-lg">
            <FileText className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-4">
            PDF to XML Converter
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-8">
            Transform your PDF documents into structured XML with AI-powered precision
          </p>
          <button 
            onClick={() => {
              const modal = document.getElementById('how-it-works-modal');
              if (modal) modal.style.display = 'flex';
            }}
            className="inline-flex items-center px-6 py-3 bg-indigo-100 hover:bg-indigo-200 text-indigo-700 rounded-full font-medium transition-all duration-200 hover:shadow-md"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            How does this work?
          </button>
        </header>

        <div className="space-y-8">
          {/* Step 1: Upload Files */}
          <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
            <div className="bg-gradient-to-r from-blue-500 to-indigo-600 px-8 py-4">
              <h2 className="text-2xl font-bold text-white flex items-center">
                <span className="bg-white bg-opacity-20 rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold mr-3">1</span>
                Upload Your PDFs
              </h2>
            </div>
            
            <div className="p-8">
              <div className="grid md:grid-cols-2 gap-4 mb-6">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="group relative overflow-hidden bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white rounded-xl px-6 py-4 font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-lg"
                >
                  <div className="flex items-center justify-center space-x-3">
                    <FileText className="w-6 h-6" />
                    <span>Select Files</span>
                  </div>
                  <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
                </button>
                
                <button
                  onClick={() => dirInputRef.current?.click()}
                  className="group relative overflow-hidden bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white rounded-xl px-6 py-4 font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-lg"
                >
                  <div className="flex items-center justify-center space-x-3">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-5L12 5H5a2 2 0 00-2 2z" />
                    </svg>
                    <span>Select Directory</span>
                  </div>
                  <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
                </button>
              </div>

              {files.length > 0 && (
                <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-800">
                      Selected Files ({files.length})
                    </h3>
                    <span className="text-sm text-gray-500">
                      Total: {(files.reduce((acc, file) => acc + file.size, 0) / 1024 / 1024).toFixed(1)} MB
                    </span>
                  </div>
                  <div className="space-y-3 max-h-64 overflow-y-auto">
                    {files.map((file) => (
                      <div key={file.name + file.lastModified} className="flex items-center justify-between bg-white rounded-lg p-4 border border-gray-100 hover:shadow-md transition-shadow">
                        <div className="flex items-center space-x-3 flex-1 min-w-0">
                          <div className="flex-shrink-0 w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                            <FileText className="w-5 h-5 text-blue-600" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 truncate" title={file.name}>
                              {file.name}
                            </p>
                            <p className="text-xs text-gray-500">
                              {(file.size / 1024 / 1024).toFixed(1)} MB
                            </p>
                          </div>
                        </div>
                        <button
                          onClick={() => removeFile(file.name)}
                          className="ml-4 p-2 text-red-500 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Step 2: Configure Instructions */}
          <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
            <div className="bg-gradient-to-r from-purple-500 to-indigo-600 px-8 py-4">
              <h2 className="text-2xl font-bold text-white flex items-center">
                <span className="bg-white bg-opacity-20 rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold mr-3">2</span>
                Configure Instructions
              </h2>
            </div>
            
            <div className="p-8">
              <p className="text-gray-600 mb-6">
                Customize how your PDF is converted. The default prompt provides a robust starting point for most documents.
              </p>
              
              <div className="space-y-4">
                <label className="block text-sm font-semibold text-gray-700">
                  XML Structure & Processing Prompt
                </label>
                <div className="relative">
                  <textarea
                    value={prompt || defaultPrompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe how the PDF should be converted to XML..."
                    className="w-full h-64 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all resize-none bg-gray-50 focus:bg-white"
                    style={{ fontFamily: 'Monaco, Consolas, monospace', fontSize: '13px' }}
                  />
                  <div className="absolute bottom-3 right-3 text-xs text-gray-400">
                    {(prompt || defaultPrompt).length} characters
                  </div>
                </div>
                <p className="text-sm text-gray-500">
                  Specify XML tags, hierarchy, citation handling, and hyperlink processing rules.
                </p>
              </div>
            </div>
          </div>

          {/* Convert Button */}
          <div className="text-center py-8">
            <button
              onClick={handleProcessFiles}
              disabled={processing || files.length === 0}
              className="group relative inline-flex items-center px-12 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-lg font-bold rounded-2xl shadow-xl hover:shadow-2xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105 disabled:hover:scale-100"
            >
              {processing ? (
                <>
                  <Loader className="w-6 h-6 mr-3 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Convert to XML
                </>
              )}
              <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-10 rounded-2xl transition-opacity duration-300"></div>
            </button>
          </div>

          {/* Results Section */}
          {(processing || results.length > 0 || error) && (
            <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
              <div className={`px-8 py-4 ${processing ? 'bg-gradient-to-r from-blue-500 to-indigo-600' : 
                results.length > 0 && !error ? 'bg-gradient-to-r from-emerald-500 to-green-600' : 
                'bg-gradient-to-r from-red-500 to-pink-600'}`}>
                <h2 className="text-2xl font-bold text-white flex items-center">
                  {processing ? (
                    <Loader className="w-6 h-6 mr-3 animate-spin" />
                  ) : results.length > 0 && !error ? (
                    <CheckCircle className="w-6 h-6 mr-3" />
                  ) : (
                    <AlertCircle className="w-6 h-6 mr-3" />
                  )}
                  {processing ? 'Processing' : results.length > 0 && !error ? 'Results' : 'Status'}
                </h2>
              </div>
              
              <div className="p-8">
                {processing && (
                  <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
                    <div className="flex items-center space-x-3">
                      <div className="flex-shrink-0">
                        <Loader className="w-8 h-8 text-blue-600 animate-spin" />
                      </div>
                      <div>
                        <p className="text-blue-800 font-semibold">Processing in progress...</p>
                        <p className="text-blue-600 text-sm">
                          {progressMessage || "Preparing to process files..."}
                        </p>
                      </div>
                    </div>
                    <div className="mt-4 bg-blue-200 rounded-full h-2">
                      <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{width: '40%'}}></div>
                    </div>
                  </div>
                )}

                {results.length > 0 && !processing && (
                  <div className="space-y-4">
                    {results.map((result, index) => (
                      <div key={index} className={`rounded-xl p-6 border ${
                        result.status === 'completed' 
                          ? 'bg-emerald-50 border-emerald-200' 
                          : 'bg-red-50 border-red-200'
                      }`}>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4">
                            <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                              result.status === 'completed' ? 'bg-emerald-100' : 'bg-red-100'
                            }`}>
                              {result.status === 'completed' ? (
                                <CheckCircle className="w-6 h-6 text-emerald-600" />
                              ) : (
                                <AlertCircle className="w-6 h-6 text-red-600" />
                              )}
                            </div>
                            <div>
                              <h3 className="font-semibold text-gray-900">
                                {result.originalFileName || result.fileName}
                              </h3>
                              {result.chunksProcessed && (
                                <p className="text-sm text-gray-600">
                                  {result.chunksProcessed} chunks processed
                                </p>
                              )}
                              {result.status === 'failed' && (
                                <p className="text-sm text-red-600 font-medium">
                                  {result.error_message || 'Conversion failed'}
                                </p>
                              )}
                            </div>
                          </div>
                          {result.status === 'completed' && (
                            <button
                              onClick={() => downloadXML(result)}
                              className="inline-flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
                            >
                              <Download className="w-4 h-4 mr-2" />
                              Download XML
                            </button>
                          )}
                        </div>
                      </div>
                    ))}

                    {results.length > 1 && results.some(r => r.status === 'completed') && (
                      <div className="pt-4 border-t border-gray-200">
                        <button
                          onClick={downloadAllXML}
                          className="w-full bg-gradient-to-r from-emerald-600 to-green-600 hover:from-emerald-700 hover:to-green-700 text-white py-3 rounded-xl font-semibold transition-all duration-200 hover:shadow-lg"
                        >
                          <Download className="w-5 h-5 inline mr-2" />
                          Download All Successful XML Files
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {error && !processing && (
                  <div className="bg-red-50 border border-red-200 rounded-xl p-6">
                    <div className="flex items-start space-x-3">
                      <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-1" />
                      <div>
                        <h3 className="text-red-800 font-semibold">Error Occurred</h3>
                        <p className="text-red-700 mt-1">{error}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Modal */}
        <div 
          id="how-it-works-modal" 
          className="fixed inset-0 bg-black bg-opacity-50 hidden items-center justify-center z-50 p-4"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              e.currentTarget.style.display = 'none';
            }
          }}
        >
          <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-8">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold text-gray-900">How It Works</h3>
                <button 
                  onClick={() => {
                    const modal = document.getElementById('how-it-works-modal');
                    if (modal) modal.style.display = 'none';
                  }}
                  className="text-gray-400 hover:text-gray-600 p-2"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-6">
                {[
                  { num: 1, title: "Upload", desc: "Select individual PDF files or an entire directory. The application lists them for review before processing.", color: "bg-blue-500" },
                  { num: 2, title: "Configure", desc: "Use the default prompt or customize the XML structure, tags, and rules (e.g., handling citations, tables, hyperlinks) via the instruction box.", color: "bg-purple-500" },
                  { num: 3, title: "Process", desc: "PDFs are chunked (e.g., by page count). Each chunk is individually processed by an LLM according to your instructions to generate XML content.", color: "bg-green-500" },
                  { num: 4, title: "Stitch", desc: "XML outputs from all processed chunks of a single PDF are combined into a final, coherent XML document for that PDF.", color: "bg-orange-500" },
                  { num: 5, title: "Download", desc: "Get structured XML files for each successfully processed PDF. Failed conversions will be clearly indicated with error details if available.", color: "bg-red-500" }
                ].map((step) => (
                  <div key={step.num} className="flex items-start space-x-4">
                    <div className={`${step.color} text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 text-sm font-bold`}>
                      {step.num}
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-1">{step.title}</h4>
                      <p className="text-gray-600 text-sm leading-relaxed">{step.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-16 mb-8">
          <p className="text-gray-400 text-sm">PDF to XML Converter v2.0.0</p>
        </footer>
      </div>
    </div>
  );
};

export default PDFToXMLConverter;