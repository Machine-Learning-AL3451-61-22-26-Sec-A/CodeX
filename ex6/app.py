import streamlit as st
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# Load the inbuilt WHO dataset
def load_data():
    data = {
        'Fever': [1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        'Cough': [1, 1, 1, 0, 1, 0, 0, 0, 1, 1],
        'Fatigue': [1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
        'CORONA': [1, 1, 1, 1, 0, 0, 1, 0, 0, 1]
    }
    return pd.DataFrame(data)

data = load_data()

# Define the Bayesian Network structure
def build_bayesian_network():
    # Define the structure of the Bayesian network
    model = BayesianModel([('Fever', 'CORONA'), ('Cough', 'CORONA'), ('Fatigue', 'CORONA')])
    # Learn the parameters using Bayesian Estimator
    model.fit(data, estimator=BayesianEstimator)
    return model

# Perform inference on the Bayesian Network
def infer_diagnosis(model, symptoms):
    # Perform inference using Variable Elimination
    infer = VariableElimination(model)
    # Compute the probability distribution of CORONA given symptoms
    query_result = infer.query(variables=['CORONA'], evidence=symptoms)
    return query_result.values[1]

# Streamlit UI
def main():
    st.title("CORONA Diagnosis using Bayesian Network")
    st.sidebar.header("Symptoms")
    # User input for symptoms
    fever = st.sidebar.checkbox("Fever")
    cough = st.sidebar.checkbox("Cough")
    fatigue = st.sidebar.checkbox("Fatigue")
    symptoms = {'Fever': int(fever), 'Cough': int(cough), 'Fatigue': int(fatigue)}

    if st.sidebar.button("Diagnose"):
        # Build Bayesian network
        model = build_bayesian_network()
        # Perform inference
        prob_corona = infer_diagnosis(model, symptoms)
        st.write("Probability of CORONA infection:", prob_corona)

if __name__ == "__main__":
    main()

