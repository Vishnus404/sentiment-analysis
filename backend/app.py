from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.model_path = "./model"
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load model from local fine-tuned weights or from HuggingFace Hub"""
        try:
            # Check if fine-tuned model exists
            if os.path.exists(self.model_path) and os.path.exists(os.path.join(self.model_path, "config.json")):
                logger.info("Loading fine-tuned model from local path...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            else:
                logger.info("Loading pre-trained model from HuggingFace Hub...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, text):
        """Perform sentiment prediction on input text"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Get prediction results
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
            
            # Map to sentiment labels
            label_map = {0: "negative", 1: "positive"}
            label = label_map[predicted_class]
            
            return {
                "label": label,
                "score": round(confidence, 4)
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

# Initialize the sentiment analyzer
analyzer = SentimentAnalyzer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "device": str(analyzer.device)})

@app.route('/predict', methods=['POST'])
def predict():
    """Sentiment prediction endpoint"""
    try:
        # Validate request
        if not request.json or 'text' not in request.json:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = request.json['text']
        
        # Validate text input
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({"error": "Text must be a non-empty string"}), 400
        
        # Perform prediction
        result = analyzer.predict(text)
        
        # Log prediction
        logger.info(f"Prediction: {text[:50]}... -> {result['label']} ({result['score']})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        info = {
            "model_name": analyzer.model_name,
            "device": str(analyzer.device),
            "model_path": analyzer.model_path,
            "is_fine_tuned": os.path.exists(analyzer.model_path) and os.path.exists(os.path.join(analyzer.model_path, "config.json"))
        }
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Reload model (useful after fine-tuning)"""
    try:
        analyzer.load_model()
        return jsonify({"message": "Model reloaded successfully"})
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        return jsonify({"error": "Failed to reload model"}), 500

if __name__ == '__main__':
    # Create model directory if it doesn't exist
    os.makedirs("./model", exist_ok=True)
    
    logger.info("Starting Sentiment Analysis API...")
    app.run(host='0.0.0.0', port=8000, debug=True)