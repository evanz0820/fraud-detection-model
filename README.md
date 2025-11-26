# Credit Card Fraud Detection System

A full-stack machine learning application for detecting fraudulent credit card transactions using PyTorch neural networks with an Angular dashboard for visualization.

## 🏗️ Architecture

```
fraud-detection/
├── backend/                 # Python Flask API + PyTorch Model
│   ├── model.py            # PyTorch fraud detection model
│   ├── app.py              # Flask REST API
│   └── requirements.txt    # Python dependencies
├── frontend/               # Angular Dashboard
│   └── src/app/
│       ├── app.component.* # Main dashboard component
│       └── services/       # API services
└── README.md
```

## 🧠 Model Details

### Dataset
- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Fraud Cases**: 492 (0.172% - highly imbalanced)
- **Features**: 30 (Time, V1-V28 PCA components, Amount)

### Neural Network Architecture
```
Input (30 features)
    ↓
Linear(30, 128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(128, 64) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(64, 32) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(32, 1) → Sigmoid
    ↓
Output (fraud probability)
```

### Techniques Used
- **SMOTE** (Synthetic Minority Over-sampling) for class imbalance
- **Batch Normalization** for training stability
- **Dropout** for regularization
- **Learning Rate Scheduling** with ReduceLROnPlateau
- **Optimal Threshold Selection** using Precision-Recall curve

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- npm or yarn

### 1. Download Dataset
Download from Kaggle and place in backend/data/:
```bash
mkdir -p backend/data
# Download creditcard.csv from Kaggle and place it here
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Train the model
python model.py data/creditcard.csv

# Start the API server
python app.py
```

The API will be available at `http://localhost:5000`

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies (use --legacy-peer-deps for compatibility)
npm install --legacy-peer-deps

# Start development server
npx ng serve
```

The dashboard will be available at `http://localhost:4200`

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/model/info` | GET | Model architecture details |
| `/api/model/metrics` | GET | Performance metrics (accuracy, precision, recall, etc.) |
| `/api/model/training-history` | GET | Training loss over epochs |
| `/api/dashboard/stats` | GET | Dashboard statistics |
| `/api/predict` | POST | Make fraud prediction |
| `/api/simulate/transaction` | GET | Simulate a transaction |

### Example Prediction Request
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0,
    "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37,
    "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61,
    "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47,
    "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
    "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06,
    "V25": 0.12, "V26": -0.18, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62
  }'
```

## 📈 Dashboard Features

1. **Overview Stats**
   - Total transactions processed
   - Fraud cases detected
   - Fraud rate percentage
   - Model accuracy

2. **Model Performance**
   - Accuracy, Precision, Recall, F1 Score, ROC-AUC
   - Confusion matrix visualization
   - Precision-Recall curve

3. **Training Analytics**
   - Training loss over epochs
   - Learning rate schedule

4. **Live Transaction Monitor**
   - Real-time transaction simulation
   - Fraud probability visualization
   - Risk level indicators

5. **Data Distribution**
   - Hourly transaction patterns
   - Risk distribution breakdown

## 🎯 Expected Model Performance

After training on the full dataset:

| Metric | Expected Value |
|--------|---------------|
| Accuracy | ~99.9% |
| Precision | ~90-95% |
| Recall | ~75-85% |
| F1 Score | ~82-90% |
| ROC-AUC | ~97-99% |

## 🔧 Configuration

### Backend Environment Variables
```bash
FLASK_ENV=development
FLASK_DEBUG=1
MODEL_DIR=models
DATA_PATH=data/creditcard.csv
```

### Frontend Environment
Edit `src/environments/environment.ts`:
```typescript
export const environment = {
  production: false,
  apiUrl: 'http://localhost:5000/api'
};
```

## 📝 Demo Mode

The application works in **demo mode** even without a trained model:
- Mock data is generated for all visualizations
- Simulated transactions show realistic fraud patterns
- All dashboard features remain functional

This is useful for frontend development or demonstrations.

## 🛠️ Development

### Running Tests
```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
ng test
```

### Building for Production
```bash
# Frontend
cd frontend
ng build --configuration production

# The build artifacts will be in dist/
```

## 📚 References

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- PyTorch: [pytorch.org](https://pytorch.org)
- Angular: [angular.io](https://angular.io)
- SMOTE: [imbalanced-learn](https://imbalanced-learn.org)

## 📄 License

MIT License - feel free to use this project for learning and development.
# fraud-detection-model
