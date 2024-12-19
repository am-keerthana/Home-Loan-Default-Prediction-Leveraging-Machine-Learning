from flask import Flask, render_template, request
import numpy as np
import pickle
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and scaler
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    model = None
    scaler = None

def validate_inputs(income, loan_amount, age, employment_length, credit_history_length):
    """Validate input values for reasonableness."""
    errors = []
    
    # Minimum income requirement (e.g., ₹1,80,000 annually - minimum wage in India)
    if income < 180000:
        errors.append("Annual income must be at least ₹1,80,000")
    
    # Maximum loan amount relative to income (e.g., 5 times annual income)
    if loan_amount > income * 5:
        errors.append("Loan amount cannot exceed 5 times your annual income")
    
    # Minimum loan amount
    if loan_amount < 10000:
        errors.append("Loan amount must be at least ₹10,000")
    
    # Age restrictions
    if age < 18:
        errors.append("Applicant must be at least 18 years old")
    if age > 100:
        errors.append("Please verify the age entered")
    
    # Employment length validation
    if employment_length > age - 18:
        errors.append("Employment length cannot be longer than (age - 18)")
    
    # Credit history length validation
    if credit_history_length > age - 18:
        errors.append("Credit history length cannot be longer than (age - 18)")
        
    return errors

def calculate_metrics(income, loan_amount):
    """Calculate financial metrics from income and loan amount."""
    debt_ratio = (loan_amount / income) * 100
    credit_to_income = loan_amount / income
    return {
        'debt_ratio': debt_ratio,
        'credit_to_income': credit_to_income
    }

def get_risk_factors(age, employment_length, credit_history_length, debt_ratio, credit_to_income):
    """Identify risk factors based on input values."""
    risk_factors = []
    
    if age < 25:
        risk_factors.append("Young age (under 25) may indicate less financial stability")
    if employment_length < 2:
        risk_factors.append("Short employment history (less than 2 years)")
    if credit_history_length < 1:
        risk_factors.append("Limited credit history (less than 1 year)")
    if debt_ratio > 50:
        risk_factors.append("High debt-to-income ratio (over 50%)")
    if credit_to_income > 3:
        risk_factors.append("Loan amount is more than 3 times annual income")
        
    return risk_factors

@app.route('/', methods=['GET', 'POST'])
def predict():
    # Initialize all variables that will be used in the template
    prediction = None
    error = None
    risk_probability = 0  # Initialize with a default value
    risk_factors = []
    metrics = None
    
    if request.method == 'POST':
        try:
            # Get input values
            income = float(request.form['income'])
            loan_amount = float(request.form['credit_amount'])
            age = float(request.form['age'])
            employment_length = float(request.form['employment_length'])
            credit_history_length = float(request.form['credit_history_length'])
            
            # Validate inputs
            validation_errors = validate_inputs(
                income, loan_amount, age, employment_length, credit_history_length
            )
            
            if validation_errors:
                error = "Validation errors: " + "; ".join(validation_errors)
                return render_template('index.html', 
                                    error=error,
                                    prediction=prediction,
                                    risk_probability=risk_probability,
                                    risk_factors=risk_factors,
                                    metrics=metrics)
            
            # Calculate financial metrics
            metrics = calculate_metrics(income, loan_amount)
            debt_ratio = metrics['debt_ratio']
            credit_to_income = metrics['credit_to_income']
            
            # Calculate risk score
            young_age_risk = 1 if age < 25 else 0
            short_employment = 1 if employment_length < 2 else 0
            limited_credit_history = 1 if credit_history_length < 1 else 0
            high_debt_ratio = 1 if debt_ratio > 50 else 0
            high_credit_amount = 1 if credit_to_income > 3 else 0
            
            risk_score = (
                (young_age_risk * 20) +
                (short_employment * 20) +
                (limited_credit_history * 20) +
                (high_debt_ratio * 20) +
                (high_credit_amount * 20)
            )
            
            # Prepare features for prediction
            features_array = np.array([[
                income,
                loan_amount,
                age,
                employment_length,
                credit_history_length,
                debt_ratio,
                credit_to_income,
                young_age_risk,
                short_employment,
                limited_credit_history,
                high_debt_ratio,
                high_credit_amount,
                risk_score
            ]])
            
            # Scale features and make prediction
            if scaler is not None and model is not None:
                features_scaled = scaler.transform(features_array)
                prediction = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0]
                risk_probability = probabilities[1] * 100
                logger.info(f"Prediction made successfully: {prediction} (Risk: {risk_probability:.2f}%)")
            else:
                error = "Model or scaler not available"
                logger.error(error)
                return render_template('index.html', 
                                    error=error,
                                    prediction=prediction,
                                    risk_probability=risk_probability,
                                    risk_factors=risk_factors,
                                    metrics=metrics)
            
            # Get risk factors
            risk_factors = get_risk_factors(
                age, employment_length, credit_history_length,
                debt_ratio, credit_to_income
            )
            
        except ValueError as ve:
            error = str(ve)
            logger.error(f"Validation error: {error}")
        except Exception as e:
            error = f"An error occurred: {str(e)}"
            logger.error(f"Prediction error: {error}")
    
    return render_template('index.html', 
                         prediction=prediction, 
                         error=error, 
                         risk_probability=risk_probability,
                         risk_factors=risk_factors,
                         metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)
