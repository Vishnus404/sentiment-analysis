import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);

  // API base URL - adjust for your environment
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  useEffect(() => {
    // Fetch model info on component mount
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/model/info`);
      setModelInfo(response.data);
    } catch (err) {
      console.error('Error fetching model info:', err);
    }
  };

  const handlePredict = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        text: text.trim()
      });

      const prediction = response.data;
      setResult(prediction);
      
      // Add to history
      setPredictionHistory(prev => [{
        text: text.trim(),
        ...prediction,
        timestamp: new Date().toLocaleTimeString()
      }, ...prev.slice(0, 4)]); // Keep only last 5 predictions
      
    } catch (err) {
      console.error('Error making prediction:', err);
      setError(
        err.response?.data?.error || 
        'Error connecting to the sentiment analysis service. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handlePredict();
    }
  };

  const getSentimentEmoji = (label) => {
    return label === 'positive' ? '😊' : '😞';
  };

  const getConfidenceColor = (score) => {
    if (score > 0.8) return '#4CAF50';
    if (score > 0.6) return '#FF9800';
    return '#F44336';
  };

  const sampleTexts = [
    "I love this product! It's amazing and works perfectly.",
    "This is terrible. I hate it and want my money back.",
    "The service was okay, nothing special but not bad either.",
    "Outstanding quality! Highly recommended to everyone.",
    "Poor customer service and low quality materials."
  ];

  return (
    <div className="App">
      <div className="container">
        <div className="header">
          <h1 className="title">🤖 Sentiment Analysis</h1>
          <p className="subtitle">
            Analyze the sentiment of any text using advanced AI models
          </p>
        </div>

        {/* Model Info */}
        {modelInfo && (
          <div className="stats">
            <div className="stat-card">
              <div className="stat-value">{modelInfo.is_fine_tuned ? '✅' : '🔄'}</div>
              <div className="stat-label">{modelInfo.is_fine_tuned ? 'Fine-tuned' : 'Pre-trained'}</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{modelInfo.device.toUpperCase()}</div>
              <div className="stat-label">Device</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{predictionHistory.length}</div>
              <div className="stat-label">Predictions</div>
            </div>
          </div>
        )}

        {/* Main Input Card */}
        <div className="card">
          <div className="form-group">
            <textarea
              className="textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Enter text to analyze sentiment... (Ctrl+Enter to predict)"
              rows="6"
            />
          </div>
          
          <button 
            className="button" 
            onClick={handlePredict}
            disabled={loading || !text.trim()}
          >
            {loading ? (
              <div className="loading">
                <div className="spinner"></div>
                Analyzing...
              </div>
            ) : (
              'Analyze Sentiment'
            )}
          </button>

          {/* Sample Texts */}
          <div style={{ marginTop: '1rem' }}>
            <p style={{ color: '#666', fontSize: '0.9rem', marginBottom: '0.5rem' }}>
              Try these examples:
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {sampleTexts.map((sample, index) => (
                <button
                  key={index}
                  onClick={() => setText(sample)}
                  style={{
                    padding: '0.3rem 0.8rem',
                    fontSize: '0.8rem',
                    background: '#f0f0f0',
                    border: 'none',
                    borderRadius: '15px',
                    cursor: 'pointer',
                    transition: 'background 0.2s'
                  }}
                  onMouseEnter={(e) => e.target.style.background = '#e0e0e0'}
                  onMouseLeave={(e) => e.target.style.background = '#f0f0f0'}
                >
                  {sample.substring(0, 30)}...
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="error">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Result Display */}
        {result && (
          <div className={`result ${result.label}`}>
            <div className="result-label">
              {getSentimentEmoji(result.label)} {result.label}
            </div>
            <div className="result-score">
              Confidence: {(result.score * 100).toFixed(1)}%
            </div>
            <div style={{ 
              marginTop: '0.5rem', 
              fontSize: '0.9rem', 
              opacity: 0.8 
            }}>
              {result.score > 0.8 ? 'Very confident' : 
               result.score > 0.6 ? 'Moderately confident' : 
               'Less confident'}
            </div>
          </div>
        )}

        {/* Prediction History */}
        {predictionHistory.length > 0 && (
          <div className="card">
            <h3 style={{ marginBottom: '1rem', color: '#333' }}>Recent Predictions</h3>
            {predictionHistory.map((pred, index) => (
              <div key={index} style={{ 
                marginBottom: '1rem', 
                padding: '1rem',
                background: '#f9f9f9',
                borderRadius: '8px',
                borderLeft: `4px solid ${pred.label === 'positive' ? '#4CAF50' : '#F44336'}`
              }}>
                <div style={{ 
                  fontSize: '0.9rem', 
                  color: '#666', 
                  marginBottom: '0.5rem' 
                }}>
                  {pred.timestamp}
                </div>
                <div style={{ 
                  fontSize: '0.95rem', 
                  marginBottom: '0.5rem',
                  fontStyle: 'italic'
                }}>
                  "{pred.text.substring(0, 100)}{pred.text.length > 100 ? '...' : ''}"
                </div>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '1rem' 
                }}>
                  <span style={{ 
                    color: pred.label === 'positive' ? '#4CAF50' : '#F44336',
                    fontWeight: 'bold'
                  }}>
                    {getSentimentEmoji(pred.label)} {pred.label.toUpperCase()}
                  </span>
                  <span style={{ 
                    color: getConfidenceColor(pred.score),
                    fontSize: '0.9rem'
                  }}>
                    {(pred.score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Footer */}
        <div style={{ 
          marginTop: '2rem', 
          padding: '1rem', 
          color: 'rgba(255,255,255,0.7)', 
          fontSize: '0.9rem' 
        }}>
          <p>
            💡 <strong>Tip:</strong> Use Ctrl+Enter to quickly analyze text
          </p>
          <p>
            🔧 Powered by Transformers and Flask | Built with React
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;