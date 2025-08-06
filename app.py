from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
import os
import json
from datetime import datetime
import pandas as pd
from werkzeug.utils import secure_filename

from claim_analyzer import ClaimAnalyzer
from document_processor import DocumentProcessor

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
analyzer = ClaimAnalyzer()
doc_processor = DocumentProcessor()

# Store analysis results
analysis_results = []

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'docx', 'doc'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the document
            text = doc_processor.process_document(filepath)
            claim_info = doc_processor.extract_claim_info(text)
            doc_quality = doc_processor.analyze_document_quality(text)
            
            return render_template('manual_input.html', 
                                 extracted_text=text,
                                 claim_info=claim_info,
                                 doc_quality=doc_quality,
                                 filename=filename)
        
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type')
    return redirect(url_for('index'))

@app.route('/manual')
def manual_input():
    return render_template('manual_input.html')

@app.route('/analyze', methods=['POST'])
def analyze_claim():
    try:
        # Get form data
        claim_data = {
            'claim_text': request.form.get('claim_text', ''),
            'claim_amount': float(request.form.get('claim_amount', 0)),
            'claimant_age': int(request.form.get('claimant_age', 30)),
            'policy_age_days': int(request.form.get('policy_age_days', 365)),
            'previous_claims': int(request.form.get('previous_claims', 0)),
            'claim_age_days': int(request.form.get('claim_age_days', 1))
        }
        
        # Validate required fields
        if not claim_data['claim_text'].strip():
            flash('Claim description is required')
            return redirect(url_for('manual_input'))
        
        if claim_data['claim_amount'] <= 0:
            flash('Claim amount must be greater than 0')
            return redirect(url_for('manual_input'))
        
        # Analyze the claim
        analysis = analyzer.analyze_claim(claim_data)
        
        # Create result record
        result = {
            'timestamp': datetime.now().isoformat(),
            'claim_data': claim_data,
            'analysis': analysis,
            'id': len(analysis_results) + 1
        }
        
        analysis_results.append(result)
        
        return render_template('results.html', 
                             claim_data=claim_data,
                             analysis=analysis,
                             result_id=result['id'])
    
    except Exception as e:
        flash(f'Error analyzing claim: {str(e)}')
        return redirect(url_for('manual_input'))

@app.route('/dashboard')
def dashboard():
    if not analysis_results:
        return render_template('dashboard.html', 
                             stats={}, 
                             recent_claims=[],
                             has_data=False)
    
    # Calculate statistics
    total_claims = len(analysis_results)
    fraud_claims = sum(1 for r in analysis_results if r['analysis']['is_likely_fraud'])
    high_risk_claims = sum(1 for r in analysis_results if r['analysis']['risk_level'] == 'High')
    approved_claims = sum(1 for r in analysis_results if r['analysis']['approval_recommendation']['decision'] == 'APPROVE')
    
    avg_fraud_prob = sum(r['analysis']['fraud_probability'] for r in analysis_results) / total_claims
    avg_risk_score = sum(r['analysis']['risk_score'] for r in analysis_results) / total_claims
    avg_claim_amount = sum(r['claim_data']['claim_amount'] for r in analysis_results) / total_claims
    
    stats = {
        'total_claims': total_claims,
        'fraud_rate': round((fraud_claims / total_claims) * 100, 1),
        'high_risk_rate': round((high_risk_claims / total_claims) * 100, 1),
        'approval_rate': round((approved_claims / total_claims) * 100, 1),
        'avg_fraud_probability': round(avg_fraud_prob, 1),
        'avg_risk_score': round(avg_risk_score, 1),
        'avg_claim_amount': round(avg_claim_amount, 2)
    }
    
    # Get recent claims (last 10)
    recent_claims = analysis_results[-10:][::-1]  # Reverse to show most recent first
    
    return render_template('dashboard.html', 
                         stats=stats,
                         recent_claims=recent_claims,
                         has_data=True)

@app.route('/api/claims')
def api_claims():
    """API endpoint to get all claims data"""
    return jsonify(analysis_results)

@app.route('/api/stats')
def api_stats():
    """API endpoint to get statistics"""
    if not analysis_results:
        return jsonify({'error': 'No data available'})
    
    # Risk level distribution
    risk_levels = {'Low': 0, 'Medium': 0, 'High': 0}
    for result in analysis_results:
        risk_level = result['analysis']['risk_level']
        risk_levels[risk_level] += 1
    
    # Decision distribution
    decisions = {'APPROVE': 0, 'REVIEW': 0, 'REJECT': 0}
    for result in analysis_results:
        decision = result['analysis']['approval_recommendation']['decision']
        decisions[decision] += 1
    
    # Claims over time (by day)
    claims_by_date = {}
    for result in analysis_results:
        date = result['timestamp'][:10]  # Get date part
        claims_by_date[date] = claims_by_date.get(date, 0) + 1
    
    return jsonify({
        'risk_distribution': risk_levels,
        'decision_distribution': decisions,
        'claims_by_date': claims_by_date
    })

@app.route('/claim/<int:claim_id>')
def view_claim(claim_id):
    """View detailed information about a specific claim"""
    if claim_id <= 0 or claim_id > len(analysis_results):
        flash('Claim not found')
        return redirect(url_for('dashboard'))
    
    result = analysis_results[claim_id - 1]
    return render_template('claim_detail.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
