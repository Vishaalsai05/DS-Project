import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
feature_columns = bundle['feature_columns']
best_threshold = bundle['best_threshold']
balance_bins = bundle['balance_bins']

CATEGORICAL_COLS = ['job', 'marital', 'education', 'default',
                     'housing', 'loan', 'contact', 'month', 'poutcome']

def group_job(job):
    if job in ['management', 'professional', 'admin.', 'technician']:
        return 'white_collar'
    elif job in ['blue-collar', 'services', 'housemaid']:
        return 'blue_collar'
    elif job in ['retired', 'student', 'unemployed']:
        return 'low_income_inactive'
    else:
        return 'self_employed'

def preprocess(input_dict):
    df = pd.DataFrame([input_dict])

    df['was_contacted_before'] = (df['pdays'] != -1).astype(int)
    df['job_tier'] = df['job'].apply(group_job)

    df['pdays_bucket'] = pd.cut(
        df['pdays'], bins=[-2, -1, 30, 90, 180, 1000],
        labels=['never', 'recent', '1-3mo', '3-6mo', '6mo+']
    )
    df['campaign_bucket'] = pd.cut(df['campaign'], bins=[0,1,3,6,100],
                                 labels=['1','2-3','4-6','7+']
    )
    df['age_bucket'] = pd.cut(
        df['age'], bins=[17, 25, 35, 45, 55, 65, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
    )
   
    df['balance_bucket'] = pd.cut(df['balance'], bins=balance_bins)


    bucket_cols = ['job_tier', 'pdays_bucket', 'campaign_bucket',
                   'balance_bucket', 'age_bucket']
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS + bucket_cols,dtype=int)

    # Strip characters XGBoost rejects in feature names
    df_encoded.columns = df_encoded.columns.str.replace(r'[\[\]<,]', '', regex=True)

    # Align to the exact column set/order the model was trained on.
    # Any dummy column not present for this input (e.g. a job category
    # that wasn't selected) gets filled with 0, matching a "not this
    # category" row from training.
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return df_encoded

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    input_dict = {
        'age': int(form['age']),
        'job': form['job'],
        'marital': form['marital'],
        'education': form['education'],
        'default': form['default'],
        'balance': int(form['balance']),
        'housing': form['housing'],
        'loan': form['loan'],
        'contact': form['contact'],
        'day': int(form['day']),
        'month': form['month'],
        'campaign': int(form['campaign']),
        'pdays': int(form['pdays']),
        'previous': int(form['previous']),
        'poutcome': form['poutcome'],
    }
 
    X = preprocess(input_dict)
    proba = float(model.predict_proba(X)[:, 1][0])
    prediction = int(proba >= best_threshold)
 
    result = {
        'label': 'Yes — likely to subscribe' if prediction == 1 else 'No — unlikely to subscribe',
        'probability': round(proba, 3),
        'threshold_used': best_threshold,
    }
    return render_template('index.html', result=result, form=input_dict)
 
 
if __name__ == '__main__':
    app.run(debug=True)