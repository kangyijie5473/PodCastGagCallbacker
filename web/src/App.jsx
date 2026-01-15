import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Upload, Link as LinkIcon, Podcast, Play, FileAudio, Loader2, Sparkles, AlertCircle, CheckCircle2, Library, ChevronDown, ChevronRight, Clock } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const API_BASE = "http://localhost:8000/api";

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

function App() {
  const [activeTab, setActiveTab] = useState('search'); // search, upload, link, library
  
  return (
    <div className="min-h-screen bg-background text-white selection:bg-secondary/30">
      <div className="max-w-5xl mx-auto px-4 py-8">
        <header className="mb-12 flex flex-col items-center text-center">
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="inline-flex items-center gap-2 mb-4 px-4 py-1.5 rounded-full bg-surface border border-white/10 text-sm font-medium text-secondary"
          >
            <Sparkles className="w-4 h-4" />
            <span>AI Powered Podcast Search</span>
          </motion.div>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6 bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent">
            PodCast Gag
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl">
            Search through your favorite podcasts with semantic understanding. 
            Upload your own audio or ingest from XiaoYuZhou.
          </p>
        </header>

        <nav className="flex justify-center mb-12">
          <div className="flex p-1 bg-surface rounded-full border border-white/10 shadow-xl overflow-x-auto">
            {[
              { id: 'search', icon: Search, label: 'Search' },
              { id: 'upload', icon: Upload, label: 'Upload' },
              { id: 'link', icon: LinkIcon, label: 'Link' },
              { id: 'library', icon: Library, label: 'Library' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "flex items-center gap-2 px-6 py-3 rounded-full text-sm font-medium transition-all duration-300 whitespace-nowrap",
                  activeTab === tab.id 
                    ? "bg-primary text-white shadow-lg shadow-primary/25" 
                    : "text-gray-400 hover:text-white hover:bg-white/5"
                )}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </nav>

        <main className="min-h-[400px]">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.2 }}
            >
              {activeTab === 'search' && <SearchView />}
              {activeTab === 'upload' && <UploadView />}
              {activeTab === 'link' && <LinkView />}
              {activeTab === 'library' && <LibraryView />}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

function SearchView() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [useRag, setUseRag] = useState(false);
  const [answer, setAnswer] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResults([]);
    setAnswer(null);

    try {
      const res = await axios.post(`${API_BASE}/search`, {
        query,
        use_rag: useRag
      });
      setResults(res.data.results);
      setAnswer(res.data.answer);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <form onSubmit={handleSearch} className="relative group">
        <div className="absolute inset-0 bg-gradient-to-r from-primary to-secondary rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-500"></div>
        <div className="relative flex items-center bg-surface border border-white/10 rounded-2xl p-2 shadow-2xl">
          <Search className="w-6 h-6 text-gray-400 ml-4" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search for moments, topics, or quotes..."
            className="flex-1 bg-transparent border-none outline-none px-4 py-3 text-lg placeholder:text-gray-500"
          />
          <button
            type="button"
            onClick={() => setUseRag(!useRag)}
            className={cn(
              "px-3 py-1.5 rounded-lg text-xs font-medium mr-2 transition-colors border",
              useRag 
                ? "bg-secondary/20 border-secondary text-secondary" 
                : "bg-white/5 border-white/10 text-gray-400 hover:text-white"
            )}
          >
            AI Answer {useRag ? 'ON' : 'OFF'}
          </button>
          <button
            type="submit"
            disabled={loading}
            className="bg-white text-black px-6 py-3 rounded-xl font-semibold hover:bg-gray-100 transition-transform active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : "Search"}
          </button>
        </div>
      </form>

      {answer && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-gradient-to-br from-surface to-surface/50 border border-secondary/20 rounded-2xl p-6 shadow-2xl relative overflow-hidden"
        >
          <div className="absolute top-0 right-0 w-32 h-32 bg-secondary/10 blur-3xl rounded-full"></div>
          <div className="flex gap-3 mb-4">
            <Sparkles className="w-5 h-5 text-secondary" />
            <h3 className="font-semibold text-lg text-secondary">AI Summary</h3>
          </div>
          <p className="text-gray-200 leading-relaxed whitespace-pre-wrap">{answer}</p>
        </motion.div>
      )}

      <div className="space-y-4">
        {results.map((item, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            className="bg-surface border border-white/5 rounded-xl p-5 hover:border-primary/30 transition-colors group cursor-pointer"
          >
            <div className="flex justify-between items-start mb-2">
              <div className="flex items-center gap-2 text-xs font-medium text-primary bg-primary/10 px-2 py-1 rounded">
                <Podcast className="w-3 h-3" />
                {item.podcast}
              </div>
              <span className="text-xs text-gray-500 font-mono">
                {formatTime(item.start)} - {formatTime(item.end)}
              </span>
            </div>
            <p className="text-gray-300 group-hover:text-white transition-colors">
              {item.text}
            </p>
            <div className="mt-3 flex items-center gap-2 text-xs text-gray-500">
              <FileAudio className="w-3 h-3" />
              <span className="truncate max-w-xs">{item.audio_id}</span>
            </div>
          </motion.div>
        ))}
        {!loading && results.length === 0 && query && !answer && (
          <div className="text-center text-gray-500 py-12">
            No results found. Try a different query.
          </div>
        )}
      </div>
    </div>
  );
}

function UploadView() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState(null);
  const [showConfirm, setShowConfirm] = useState(false);

  useEffect(() => {
    if (!taskId) return;

    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE}/tasks/${taskId}`);
        setTaskStatus(res.data);
        if (res.data.status === 'completed' || res.data.status === 'error') {
          clearInterval(interval);
          setUploading(false);
        }
      } catch (err) {
        console.error("Polling error", err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [taskId]);

  const handleFileSelect = (e) => {
    const selected = e.target.files[0];
    if (selected) {
        setFile(selected);
        setShowConfirm(true);
    }
  };

  const getEstimate = (size) => {
      if (!size) return "Unknown";
      const sizeMB = size / (1024 * 1024);
      // Rough estimate: 1MB ~ 12s processing (based on 0.2 RTF for 128kbps audio)
      const estSeconds = Math.ceil(sizeMB * 12); 
      if (estSeconds < 60) return `${estSeconds} seconds`;
      return `${Math.ceil(estSeconds / 60)} minutes`;
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setShowConfirm(false);
    setTaskId(null);
    setTaskStatus(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setTaskId(res.data.task_id);
    } catch (err) {
      console.error(err);
      setUploading(false);
      setTaskStatus({ status: 'error', message: 'Upload request failed' });
    }
  };

  const resetUpload = () => {
      setFile(null);
      setShowConfirm(false);
      setTaskId(null);
      setTaskStatus(null);
  };

  return (
    <div className="max-w-xl mx-auto">
      <div className="bg-surface border border-white/10 rounded-3xl p-8 text-center border-dashed border-2 hover:border-primary/50 transition-colors">
        <div className="w-16 h-16 bg-surface rounded-full flex items-center justify-center mx-auto mb-6 border border-white/5">
          <Upload className="w-8 h-8 text-primary" />
        </div>
        <h3 className="text-xl font-semibold mb-2">Upload Audio File</h3>
        <p className="text-gray-400 mb-8 text-sm">
          Support MP3, WAV, M4A. File will be processed in background.
        </p>

        {!taskStatus && (
            <>
                <input
                type="file"
                id="file-upload"
                className="hidden"
                onChange={handleFileSelect}
                accept="audio/*"
                />

                {!file ? (
                <label
                    htmlFor="file-upload"
                    className="inline-flex cursor-pointer items-center justify-center px-6 py-3 rounded-xl bg-white text-black font-semibold hover:bg-gray-100 transition-transform active:scale-95"
                >
                    Select File
                </label>
                ) : (
                <div className="space-y-6">
                    <div className="bg-white/5 rounded-xl p-4 text-left space-y-3">
                        <div className="flex justify-between items-start">
                            <div>
                                <p className="font-medium text-white truncate max-w-[200px]">{file.name}</p>
                                <p className="text-xs text-gray-400">{(file.size / (1024*1024)).toFixed(2)} MB</p>
                            </div>
                            <button onClick={resetUpload} className="text-gray-500 hover:text-white p-1">✕</button>
                        </div>
                        
                        <div className="flex items-center gap-2 text-sm text-secondary bg-secondary/10 p-2 rounded-lg">
                            <Clock className="w-4 h-4" />
                            <span>Estimated processing time: <span className="font-bold">{getEstimate(file.size)}</span></span>
                        </div>
                    </div>

                    <button
                    onClick={handleUpload}
                    disabled={uploading}
                    className="w-full py-3 rounded-xl bg-primary text-white font-semibold hover:bg-primary/90 transition-all disabled:opacity-50 shadow-lg shadow-primary/25"
                    >
                    {uploading ? <Loader2 className="w-5 h-5 animate-spin mx-auto" /> : "Confirm & Start Indexing"}
                    </button>
                </div>
                )}
            </>
        )}

        {taskStatus && (
            <div className="mt-8 text-left space-y-4">
                 <div className="flex justify-between items-center text-sm mb-1">
                    <span className="text-gray-400">{taskStatus.message}</span>
                    <span className="font-mono text-primary">{Math.round(taskStatus.progress)}%</span>
                 </div>
                 <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                    <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${taskStatus.progress}%` }}
                        className={cn("h-full rounded-full transition-all duration-500", 
                            taskStatus.status === 'error' ? 'bg-red-500' : 'bg-secondary'
                        )}
                    />
                 </div>
                 
                 {taskStatus.status === 'completed' && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center justify-center gap-2 text-green-400 pt-4">
                        <CheckCircle2 className="w-5 h-5" />
                        <span>Processing Complete! Ready to search.</span>
                        <button onClick={() => {setTaskStatus(null); setFile(null);}} className="text-xs underline ml-2 text-gray-500">Upload another</button>
                    </motion.div>
                 )}

                 {taskStatus.status === 'error' && (
                     <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center justify-center gap-2 text-red-400 pt-4">
                        <AlertCircle className="w-5 h-5" />
                        <span>Error: {taskStatus.message}</span>
                        <button onClick={() => {setTaskStatus(null);}} className="text-xs underline ml-2 text-gray-500">Try again</button>
                    </motion.div>
                 )}
            </div>
        )}
      </div>
    </div>
  );
}

function LinkView() {
  const [url, setUrl] = useState('');
  const [limit, setLimit] = useState(1);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url) return;

    setLoading(true);
    setStatus(null);

    try {
      await axios.post(`${API_BASE}/podcast/submit`, {
        url,
        limit: parseInt(limit)
      });
      setStatus('success');
      setUrl('');
    } catch (err) {
      console.error(err);
      setStatus('error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto">
       <div className="bg-surface border border-white/10 rounded-3xl p-8">
        <div className="flex items-center gap-4 mb-8">
          <div className="w-12 h-12 rounded-xl bg-secondary/10 flex items-center justify-center text-secondary">
            <LinkIcon className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-xl font-semibold">Import from XiaoYuZhou</h3>
            <p className="text-sm text-gray-400">Paste a podcast URL to ingest episodes</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">Podcast URL</label>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://www.xiaoyuzhoufm.com/podcast/..."
              className="w-full bg-black/20 border border-white/10 rounded-xl px-4 py-3 focus:border-secondary focus:ring-1 focus:ring-secondary outline-none transition-all"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">Episodes Limit</label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="1"
                max="10"
                value={limit}
                onChange={(e) => setLimit(e.target.value)}
                className="flex-1 accent-secondary"
              />
              <span className="w-12 text-center font-mono bg-white/5 rounded px-2 py-1">{limit}</span>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-secondary to-purple-600 text-white font-semibold hover:opacity-90 transition-opacity disabled:opacity-50"
          >
             {loading ? <Loader2 className="w-5 h-5 animate-spin mx-auto" /> : "Start Import"}
          </button>
        </form>

        {status === 'success' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-6 flex items-center justify-center gap-2 text-green-400">
            <CheckCircle2 className="w-5 h-5" />
            <span>Import started! Episodes are downloading.</span>
          </motion.div>
        )}
        {status === 'error' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-6 flex items-center justify-center gap-2 text-red-400">
            <AlertCircle className="w-5 h-5" />
            <span>Import failed. Please check the URL.</span>
          </motion.div>
        )}
       </div>
    </div>
  );
}

function LibraryView() {
    const [podcasts, setPodcasts] = useState([]);
    const [loading, setLoading] = useState(true);
    const [expanded, setExpanded] = useState({});

    useEffect(() => {
        fetchLibrary();
    }, []);

    const fetchLibrary = async () => {
        try {
            const res = await axios.get(`${API_BASE}/podcasts`);
            setPodcasts(res.data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const toggleExpand = (name) => {
        setExpanded(prev => ({ ...prev, [name]: !prev[name] }));
    };

    if (loading) {
        return <div className="flex justify-center py-20"><Loader2 className="w-8 h-8 animate-spin text-primary" /></div>;
    }

    if (podcasts.length === 0) {
        return <div className="text-center text-gray-500 py-20">No podcasts indexed yet.</div>;
    }

    return (
        <div className="max-w-3xl mx-auto space-y-6">
            <h2 className="text-2xl font-bold mb-6">Indexed Podcasts</h2>
            {podcasts.map((podcast) => (
                <div key={podcast.name} className="bg-surface border border-white/5 rounded-xl overflow-hidden">
                    <button 
                        onClick={() => toggleExpand(podcast.name)}
                        className="w-full flex items-center justify-between p-4 hover:bg-white/5 transition-colors"
                    >
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                                <Podcast className="w-5 h-5" />
                            </div>
                            <div className="text-left">
                                <h3 className="font-semibold">{podcast.name}</h3>
                                <p className="text-xs text-gray-400">{podcast.episodes.length} Episodes</p>
                            </div>
                        </div>
                        {expanded[podcast.name] ? <ChevronDown className="w-5 h-5 text-gray-500" /> : <ChevronRight className="w-5 h-5 text-gray-500" />}
                    </button>
                    
                    <AnimatePresence>
                        {expanded[podcast.name] && (
                            <motion.div
                                initial={{ height: 0 }}
                                animate={{ height: 'auto' }}
                                exit={{ height: 0 }}
                                className="overflow-hidden"
                            >
                                <div className="border-t border-white/5 bg-black/20">
                                    {podcast.episodes.map((ep) => (
                                        <div key={ep.id} className="flex items-center gap-3 p-3 pl-16 border-b border-white/5 last:border-0 hover:bg-white/5">
                                            <FileAudio className="w-4 h-4 text-gray-500" />
                                            <div className="flex-1 min-w-0">
                                                <p className="text-sm truncate text-gray-300">{ep.id}</p>
                                            </div>
                                            <span className="text-xs text-gray-600 font-mono">Indexed</span>
                                        </div>
                                    ))}
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            ))}
        </div>
    );
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default App;
