import React, { useState } from 'react';
import axios from 'axios';
import { Search, Clock, Cpu, BookOpen, AlertCircle } from 'lucide-react';

function App() {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await axios.post('http://127.0.0.1:8000/api/v1/query', {
        prompt: prompt
      });
      setResponse(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to communicate with RAG backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '900px', margin: '40px auto', fontFamily: 'Arial, sans-serif', padding: '0 20px' }}>
      <h2>Enterprise Knowledge Base Search & Inference Engine</h2>
      <p style={{ color: '#666' }}>Hybrid Search (BM25 + Qdrant Vectors) with Cross-Encoder Reranking</p>

      <form onSubmit={handleSearch} style={{ display: 'flex', gap: '10px', marginBottom: '30px' }}>
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Ask a question about your enterprise documents..."
          style={{ flex: 1, padding: '12px', fontSize: '16px', borderRadius: '6px', border: '1px solid #ccc' }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{ padding: '12px 24px', fontSize: '16px', backgroundColor: '#0066cc', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer' }}
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {error && (
        <div style={{ padding: '15px', backgroundColor: '#ffe6e6', color: '#cc0000', borderRadius: '6px', marginBottom: '20px' }}>
          <AlertCircle size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
          {error}
        </div>
      )}

      {response && (
        <div>
          {/* Metrics Banner */}
          <div style={{ display: 'flex', gap: '20px', padding: '15px', backgroundColor: '#f4f6f8', borderRadius: '6px', marginBottom: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Clock size={18} color="#555" />
              <span><strong>Latency:</strong> {response.metrics.latency_ms} ms</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Cpu size={18} color="#555" />
              <span><strong>Token Usage:</strong> ~{response.metrics.estimated_tokens} tokens</span>
            </div>
          </div>

          {/* Answer Box */}
          <div style={{ border: '1px solid #e0e0e0', padding: '20px', borderRadius: '6px', marginBottom: '30px' }}>
            <h3 style={{ marginTop: 0 }}>Generated Response</h3>
            <p style={{ lineHeight: '1.6', color: '#333' }}>{response.answer}</p>
          </div>

          {/* Sources Accordion */}
          <h3>Reranked Retrieved Sources</h3>
          {response.retrieved_sources.map((src, index) => (
            <div key={index} style={{ borderLeft: '4px solid #0066cc', backgroundColor: '#fafafa', padding: '15px', marginBottom: '15px', borderRadius: '0 6px 6px 0' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontWeight: 'bold' }}>Source #{index + 1}</span>
                <span style={{ fontSize: '12px', color: '#666', backgroundColor: '#eef', padding: '2px 8px', borderRadius: '4px' }}>
                  Cross-Encoder Score: {src.cross_encoder_score?.toFixed(4)}
                </span>
              </div>
              <p style={{ margin: 0, fontSize: '14px', color: '#444' }}>{src.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;