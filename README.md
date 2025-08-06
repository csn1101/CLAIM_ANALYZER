# Insurance Claim Analyzer

A comprehensive AI-powered insurance claim analysis system that uses machine learning to detect fraud, assess risk, and provide approval recommendations for insurance claims.

## Features

### 🤖 AI-Powered Analysis
- **Fraud Detection**: Uses Random Forest classifier to identify potentially fraudulent claims
- **Risk Assessment**: Employs Isolation Forest for anomaly detection and risk scoring
- **Text Analysis**: Natural language processing to extract insights from claim descriptions
- **Approval Recommendations**: Automated decision support with confidence scoring

### 📄 Document Processing
- **Multi-format Support**: PDF, Word documents, images, and text files
- **OCR Capability**: Extract text from images using Tesseract
- **Auto-extraction**: Automatically identifies claim numbers, amounts, dates, and key information
- **Quality Assessment**: Analyzes document completeness and identifies potential issues

### 📊 Analytics Dashboard
- **Real-time Statistics**: Fraud rates, risk distribution, approval rates
- **Visual Charts**: Interactive charts showing claim patterns and trends
- **Claim History**: Detailed view of all analyzed claims
- **Performance Metrics**: System accuracy and confidence scores

### 🔍 Business Logic
- **Smart Thresholds**: Configurable risk and fraud thresholds
- **Review Flagging**: Automatically flags claims requiring human review
- **Contextual Decisions**: Considers claim amount, claimant history, and policy age
- **Detailed Reporting**: Comprehensive analysis reports with reasoning

## Technology Stack

- **Backend**: Python Flask
- **Machine Learning**: Scikit-learn (Random Forest, Isolation Forest)
- **Text Processing**: NLTK, TextBlob, TF-IDF Vectorization
- **Document Processing**: PyPDF2, pdfplumber, pytesseract, python-docx
- **Frontend**: Bootstrap 5, Chart.js
- **Data Processing**: Pandas, NumPy

## Installation

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR (for image text extraction)

### Setup Steps

1. **Clone/Download the project**
   ```bash
   cd "d:\Hackathons\vibe-coding\vibe app\claim_analyzer"
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR** (Optional, for image processing)
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Add tesseract to your PATH or update the pytesseract configuration

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and navigate to: http://localhost:5000

## Usage Guide

### 1. Document Upload
- Click "Upload Document" on the home page
- Supported formats: PDF, Word (.docx), Images (PNG, JPG), Text files
- The system will automatically extract text and attempt to identify key claim information

### 2. Manual Entry
- Use "Manual Entry" for direct claim information input
- Fill in claim description, amount, and claimant details
- The system provides real-time validation and suggestions

### 3. Analysis Results
The system provides comprehensive analysis including:
- **Fraud Probability**: 0-100% likelihood of fraudulent activity
- **Risk Score**: Overall risk assessment (Low/Medium/High)
- **Approval Recommendation**: APPROVE/REVIEW/REJECT with reasoning
- **Confidence Score**: System confidence in the analysis

### 4. Dashboard
- View analytics and trends across all analyzed claims
- Monitor fraud rates and risk distributions
- Access detailed claim history and reports

## Business Logic

### Fraud Detection Model
The system uses a Random Forest classifier trained on:
- Text pattern analysis (suspicious keywords, phrases)
- Claim amount anomalies
- Claimant behavior patterns
- Temporal factors (time since incident, policy age)

### Risk Assessment
Risk scoring considers:
- **Claim Amount**: Higher amounts increase risk score
- **Previous Claims**: Frequent claimants receive higher risk scores
- **Policy Age**: New policies are considered higher risk
- **Anomaly Detection**: Statistical outliers in claim patterns

### Approval Logic
- **Auto-Approve**: Low fraud probability (<30%) and low risk score (<40)
- **Manual Review**: Medium risk cases or high-value claims with moderate fraud probability
- **Reject**: High fraud probability (>70%) or extremely suspicious patterns

## Customization

### Adjusting Thresholds
Edit the `get_approval_recommendation()` method in `claim_analyzer.py`:
```python
# High fraud probability threshold
if fraud_prob > 0.7:  # Adjust this value
    return {'decision': 'REJECT', ...}

# Risk score threshold
if risk_score > 70:  # Adjust this value
    return {'decision': 'REVIEW', ...}
```

### Adding New Features
1. **Custom Keywords**: Update fraud/legitimate keyword lists in `generate_sample_data()`
2. **New Risk Factors**: Add additional features to the `extract_features()` method
3. **Business Rules**: Modify approval logic in `get_approval_recommendation()`

## API Endpoints

- `GET /`: Home page
- `POST /upload`: Upload and process documents
- `GET /manual`: Manual claim entry form
- `POST /analyze`: Analyze claim data
- `GET /dashboard`: Analytics dashboard
- `GET /api/claims`: JSON API for all claims
- `GET /api/stats`: JSON API for statistics
- `GET /claim/<id>`: Detailed claim view

## Sample Data

The system automatically generates synthetic training data on first run:
- 1000 sample claims with realistic patterns
- 20% fraud rate for balanced training
- Various risk levels and claim types
- Text descriptions with relevant keywords

## Performance Considerations

- **Model Training**: Initial setup creates and trains models (runs once)
- **File Processing**: Large documents may take longer to process
- **Memory Usage**: Vectorized text features require adequate RAM
- **Scalability**: For production use, consider model persistence and caching

## Troubleshooting

### Common Issues

1. **Tesseract not found**
   - Install Tesseract OCR and add to PATH
   - Or update pytesseract configuration in `document_processor.py`

2. **Import errors**
   - Ensure all requirements are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **File upload fails**
   - Check file size (max 16MB)
   - Verify file format is supported
   - Ensure uploads directory has write permissions

4. **Models not training**
   - Check available memory
   - Verify scikit-learn installation
   - Review error logs for specific issues

## Future Enhancements

- **Advanced NLP**: Integration with transformer models for better text understanding
- **External Data**: Integration with external fraud databases and watchlists
- **Real-time Processing**: Stream processing for high-volume claim analysis
- **Multi-language Support**: Analysis of claims in multiple languages
- **Explainable AI**: More detailed explanations of model decisions
- **Integration APIs**: REST APIs for integration with existing claim systems

## Security Considerations

- Input validation and sanitization
- File upload restrictions and scanning
- Data encryption for sensitive information
- Access controls and user authentication
- Audit logging for all claim decisions

## License

This project is for educational and demonstration purposes. Please ensure compliance with data protection regulations when processing real insurance claims.

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments and documentation
3. Test with sample data to verify functionality
