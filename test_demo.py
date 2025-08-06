"""
Test script to demonstrate the Insurance Claim Analyzer functionality
"""

from claim_analyzer import ClaimAnalyzer
from document_processor import DocumentProcessor
import pandas as pd

def test_claim_analyzer():
    """Test the claim analyzer with sample data"""
    print("=" * 60)
    print("INSURANCE CLAIM ANALYZER - TEST DEMONSTRATION")
    print("=" * 60)
    
    # Initialize the analyzer
    print("\n1. Initializing the Claim Analyzer...")
    analyzer = ClaimAnalyzer()
    print("✓ Models loaded and ready for analysis")
    
    # Test cases - different types of claims
    test_claims = [
        {
            'name': 'Low Risk Auto Accident',
            'data': {
                'claim_text': 'Standard auto accident occurred at intersection. Police report filed, witness statements collected. Damage to front bumper and headlight. Repair estimate from certified shop attached. Driver has clean record.',
                'claim_amount': 3500.00,
                'claimant_age': 35,
                'policy_age_days': 730,
                'previous_claims': 0,
                'claim_age_days': 2
            }
        },
        {
            'name': 'Suspicious High-Value Claim',
            'data': {
                'claim_text': 'Emergency situation, total loss occurred during midnight. No witnesses available, cash payment preferred. Paperwork was lost in the incident. Need immediate processing for urgent financial situation.',
                'claim_amount': 25000.00,
                'claimant_age': 28,
                'policy_age_days': 45,
                'previous_claims': 3,
                'claim_age_days': 15
            }
        },
        {
            'name': 'Medium Risk Water Damage',
            'data': {
                'claim_text': 'Water damage from burst pipe in basement. Damage includes flooring, drywall, and furniture. Repair estimates obtained from contractors. Photos of damage available.',
                'claim_amount': 8500.00,
                'claimant_age': 45,
                'policy_age_days': 1095,
                'previous_claims': 1,
                'claim_age_days': 5
            }
        },
        {
            'name': 'Fire Damage with Documentation',
            'data': {
                'claim_text': 'House fire caused by electrical issue. Fire department report available. Insurance adjuster visited property. Repair estimates from certified contractors. Temporary housing receipts included.',
                'claim_amount': 15000.00,
                'claimant_age': 52,
                'policy_age_days': 2190,
                'previous_claims': 0,
                'claim_age_days': 7
            }
        }
    ]
    
    print("\n2. Analyzing Sample Claims...")
    print("-" * 60)
    
    results = []
    for i, test_case in enumerate(test_claims, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print(f"Claim Amount: ${test_case['data']['claim_amount']:,.2f}")
        
        # Analyze the claim
        analysis = analyzer.analyze_claim(test_case['data'])
        
        # Display results
        print(f"Fraud Probability: {analysis['fraud_probability']:.1f}%")
        print(f"Risk Level: {analysis['risk_level']}")
        print(f"Risk Score: {analysis['risk_score']:.1f}/100")
        print(f"Decision: {analysis['approval_recommendation']['decision']}")
        print(f"Reason: {analysis['approval_recommendation']['reason']}")
        print(f"Confidence: {analysis['confidence_score']:.1f}%")
        
        # Store results
        results.append({
            'case': test_case['name'],
            'amount': test_case['data']['claim_amount'],
            'fraud_prob': analysis['fraud_probability'],
            'risk_level': analysis['risk_level'],
            'decision': analysis['approval_recommendation']['decision'],
            'confidence': analysis['confidence_score']
        })
        
        print("-" * 40)
    
    # Summary statistics
    print("\n3. Analysis Summary")
    print("-" * 60)
    df = pd.DataFrame(results)
    
    print(f"Total Claims Analyzed: {len(results)}")
    print(f"Average Fraud Probability: {df['fraud_prob'].mean():.1f}%")
    print(f"Decisions: {df['decision'].value_counts().to_dict()}")
    print(f"Risk Levels: {df['risk_level'].value_counts().to_dict()}")
    print(f"Average Confidence: {df['confidence'].mean():.1f}%")
    
    # Business impact
    total_amount = df['amount'].sum()
    approved_amount = df[df['decision'] == 'APPROVE']['amount'].sum()
    rejected_amount = df[df['decision'] == 'REJECT']['amount'].sum()
    review_amount = df[df['decision'] == 'REVIEW']['amount'].sum()
    
    print(f"\n4. Business Impact")
    print("-" * 60)
    print(f"Total Claim Value: ${total_amount:,.2f}")
    print(f"Auto-Approved: ${approved_amount:,.2f} ({approved_amount/total_amount*100:.1f}%)")
    print(f"Requires Review: ${review_amount:,.2f} ({review_amount/total_amount*100:.1f}%)")
    print(f"Rejected: ${rejected_amount:,.2f} ({rejected_amount/total_amount*100:.1f}%)")
    
    return results

def test_document_processor():
    """Test document processing functionality"""
    print("\n5. Document Processing Test")
    print("-" * 60)
    
    processor = DocumentProcessor()
    
    # Test text analysis
    sample_text = """
    INSURANCE CLAIM FORM
    Claim Number: CLM123456
    Policy Number: POL789012
    Date of Incident: 03/15/2024
    Claimant: John Smith
    Incident Type: Auto Accident
    Damage Amount: $4,500.00
    
    Description: Vehicle collision at Main Street intersection. Police report filed.
    Other driver ran red light. Witness statements available. Damage to front end
    and driver side door. Vehicle towed to certified repair shop. Medical attention
    not required.
    """
    
    print("Extracting structured information from sample text...")
    claim_info = processor.extract_claim_info(sample_text)
    
    print("\nExtracted Information:")
    for key, value in claim_info.items():
        if value:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Test document quality analysis
    quality = processor.analyze_document_quality(sample_text)
    print(f"\nDocument Quality Analysis:")
    print(f"  Quality Score: {quality['quality_score']}/100")
    print(f"  Word Count: {quality['word_count']}")
    print(f"  Issues Found: {len(quality['issues'])}")
    if quality['issues']:
        for issue in quality['issues']:
            print(f"    - {issue}")

if __name__ == "__main__":
    try:
        # Run tests
        results = test_claim_analyzer()
        test_document_processor()
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nTo start the web application, run:")
        print("python app.py")
        print("\nThen open your browser and go to:")
        print("http://localhost:5000")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("Please check that all requirements are installed:")
        print("pip install -r requirements.txt")
