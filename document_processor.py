import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
import docx
import re
import logging
from datetime import datetime

class DocumentProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF using multiple methods"""
        text = ""
        
        try:
            # Method 1: PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            self.logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Method 2: pdfplumber (better for complex layouts)
        if not text.strip():
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                self.logger.warning(f"pdfplumber extraction failed: {e}")
        
        return text.strip()
    
    def extract_text_from_docx(self, file_path):
        """Extract text from Word documents"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error extracting text from DOCX: {e}")
            return ""
    
    def extract_text_from_image(self, file_path):
        """Extract text from images using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def process_document(self, file_path, file_type=None):
        """Process document and extract text based on file type"""
        if file_type is None:
            file_type = file_path.lower().split('.')[-1]
        
        text = ""
        
        if file_type in ['pdf']:
            text = self.extract_text_from_pdf(file_path)
        elif file_type in ['docx', 'doc']:
            text = self.extract_text_from_docx(file_path)
        elif file_type in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
            text = self.extract_text_from_image(file_path)
        else:
            # Try to read as plain text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        text = f.read()
                except Exception as e:
                    self.logger.error(f"Could not process file {file_path}: {e}")
        
        return text
    
    def extract_claim_info(self, text):
        """Extract structured information from claim text"""
        claim_info = {
            'claim_number': None,
            'incident_date': None,
            'claim_amount': None,
            'claimant_name': None,
            'policy_number': None,
            'incident_type': None,
            'location': None
        }
        
        # Extract claim number
        claim_patterns = [
            r'claim\s*(?:number|#|id)?\s*:?\s*([A-Z0-9\-]+)',
            r'claim\s*([A-Z0-9\-]{6,})',
            r'reference\s*(?:number|#)?\s*:?\s*([A-Z0-9\-]+)'
        ]
        
        for pattern in claim_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                claim_info['claim_number'] = match.group(1)
                break
        
        # Extract dates
        date_patterns = [
            r'incident\s*date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'date\s*of\s*(?:loss|incident)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'occurred\s*on\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                claim_info['incident_date'] = match.group(1)
                break
        
        # Extract amounts
        amount_patterns = [
            r'claim\s*amount\s*:?\s*\$?([\d,]+\.?\d*)',
            r'damage\s*(?:amount|cost)\s*:?\s*\$?([\d,]+\.?\d*)',
            r'total\s*(?:loss|amount)\s*:?\s*\$?([\d,]+\.?\d*)'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    claim_info['claim_amount'] = float(amount_str)
                    break
                except ValueError:
                    continue
        
        # Extract policy number
        policy_patterns = [
            r'policy\s*(?:number|#)?\s*:?\s*([A-Z0-9\-]+)',
            r'policy\s*([A-Z0-9\-]{6,})'
        ]
        
        for pattern in policy_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                claim_info['policy_number'] = match.group(1)
                break
        
        # Extract incident type
        incident_types = [
            'auto accident', 'car accident', 'vehicle accident',
            'fire', 'theft', 'burglary', 'water damage', 'flood',
            'vandalism', 'hail damage', 'wind damage', 'storm damage',
            'personal injury', 'slip and fall', 'medical malpractice'
        ]
        
        text_lower = text.lower()
        for incident_type in incident_types:
            if incident_type in text_lower:
                claim_info['incident_type'] = incident_type
                break
        
        return claim_info
    
    def analyze_document_quality(self, text):
        """Analyze the quality and completeness of the document"""
        quality_score = 0
        issues = []
        
        # Check text length
        if len(text) < 100:
            issues.append("Document appears to be very short")
        else:
            quality_score += 20
        
        # Check for key sections
        required_sections = [
            'incident', 'damage', 'date', 'amount', 'policy'
        ]
        
        text_lower = text.lower()
        for section in required_sections:
            if section in text_lower:
                quality_score += 15
            else:
                issues.append(f"Missing '{section}' information")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'no receipt', r'cash only', r'lost paperwork',
            r'emergency situation', r'immediate payment',
            r'under pressure', r'time sensitive'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Suspicious phrase detected: {pattern}")
                quality_score -= 10
        
        # Check for supporting documentation mentions
        supporting_docs = [
            'police report', 'medical record', 'receipt', 'estimate',
            'witness statement', 'photo', 'repair bill'
        ]
        
        doc_count = 0
        for doc_type in supporting_docs:
            if doc_type in text_lower:
                doc_count += 1
                quality_score += 5
        
        if doc_count == 0:
            issues.append("No supporting documentation mentioned")
        
        return {
            'quality_score': min(100, max(0, quality_score)),
            'issues': issues,
            'word_count': len(text.split()),
            'character_count': len(text)
        }
