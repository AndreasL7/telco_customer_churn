import gc
import requests
import streamlit as st
from streamlit_lottie import st_lottie
import time
from joblib import load

import pandas as pd

# @st.cache_data
def load_lottie_url(url: str):

    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load the model
# @st.cache_resource 
def load_model_xgb():
    primary_path = 'streamlit_gallery/utils/best_model_telco_churn.joblib'
    alternative_path = '../../utils/best_model_telco_churn.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Model not found in both primary and alternative directories!")

# Load the pipeline
# @st.cache_resource
def load_pipeline_xgb():
    primary_path = 'streamlit_gallery/utils/best_pipeline_telco_churn.joblib'
    alternative_path = '../../utils/best_pipeline_telco_churn.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Pipeline not found in both primary and alternative directories!")

# Load the model
# @st.cache_resource 
def load_model_logreg():
    primary_path = 'streamlit_gallery/utils/best_model_telco_churn_logreg.joblib'
    alternative_path = '../../utils/best_model_telco_churn_logreg.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Model not found in both primary and alternative directories!")

# Load the pipeline
# @st.cache_resource
def load_pipeline_logreg():
    primary_path = 'streamlit_gallery/utils/best_pipeline_telco_churn_logreg.joblib'
    alternative_path = '../../utils/best_pipeline_telco_churn_logreg.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Pipeline not found in both primary and alternative directories!")
        
# Load the model
# @st.cache_resource 
def load_model_svc():
    primary_path = 'streamlit_gallery/utils/best_model_telco_churn_svc.joblib'
    alternative_path = '../../utils/best_model_telco_churn_svc.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Model not found in both primary and alternative directories!")

# Load the pipeline
# @st.cache_resource
def load_pipeline_svc():
    primary_path = 'streamlit_gallery/utils/best_pipeline_telco_churn_svc.joblib'
    alternative_path = '../../utils/best_pipeline_telco_churn_svc.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Pipeline not found in both primary and alternative directories!")

# Load the model
# @st.cache_resource 
def load_model_soft():
    primary_path = 'streamlit_gallery/utils/best_model_telco_churn_voting_soft.joblib'
    alternative_path = '../../utils/best_model_telco_churn_voting_soft.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Model not found in both primary and alternative directories!")
        
# Load the model
# @st.cache_resource 
def load_model_hard():
    primary_path = 'streamlit_gallery/utils/best_model_telco_churn_voting_hard.joblib'
    alternative_path = '../../utils/best_model_telco_churn_voting_hard.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Model not found in both primary and alternative directories!")

def get_session_value(key, default_value):
    
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]

def session_slider_int(label, min_value, max_value, key, default_value, step=1):

    value = get_session_value(key, default_value)
    new_value = st.slider(label, min_value, max_value, value, step=step)
    st.session_state[key] = new_value
    return new_value

def session_slider_float(label, min_value, max_value, key, default_value, step=0.1):

    value = get_session_value(key, default_value)
    new_value = st.slider(label, min_value, max_value, value, step=step)
    st.session_state[key] = new_value
    return new_value

def session_radio(label, options, key, default_value):

    value = get_session_value(key, default_value)
    new_value = st.radio(label, options, index=options.index(value))
    st.session_state[key] = new_value
    return new_value

def session_selectbox(label, options, key, default_value):

    value = get_session_value(key, default_value)
    new_value = st.selectbox(label, options, index=options.index(value))
    st.session_state[key] = new_value
    return new_value

def session_number_input(label, key, default_value, **kwargs):

    value = get_session_value(key, default_value)
    new_value = st.number_input(label, value=value, **kwargs)
    st.session_state[key] = new_value
    return new_value
    
# Define callback function to update form content
def set_form_content(option):
    st.session_state["form_content"] = option
    
def update_form_values(new_values):
    for key, value in new_values.items():
        st.session_state[key] = value

# @st.cache_data
def make_prediction(inputs):
    
    optimal_threshold = 0.3535353535353536
    
    tweak_inputs_xgb = load_pipeline_xgb().transform(pd.DataFrame([inputs]))
    y_prob_xgb = load_model_xgb().predict_proba(tweak_inputs_xgb)[:,1]
    y_pred_xgb = (y_prob_xgb >= optimal_threshold).astype(int)
    
    tweak_inputs_logreg = load_pipeline_logreg().transform(pd.DataFrame([inputs]))
    y_prob_logreg = load_model_logreg().predict_proba(tweak_inputs_logreg)[:,1]
    y_pred_logreg = (y_prob_logreg >= optimal_threshold).astype(int)
    
    tweak_inputs_svc = load_pipeline_svc().transform(pd.DataFrame([inputs]))
    y_prob_svc = load_model_svc().predict_proba(tweak_inputs_svc)[:,1]
    y_pred_svc = (y_prob_svc >= optimal_threshold).astype(int)
    
    y_prob_soft = load_model_soft().predict_proba(tweak_inputs_logreg)[:,1]
    y_pred_soft = (y_prob_soft >= optimal_threshold).astype(int)
    y_prob_hard = load_model_hard().predict(tweak_inputs_logreg)

    return y_pred_soft[0]

def main():
    
    gc.enable()

    st.title("Is this customer displaying sign of churning?")
    st.subheader("Catch them before they leave!")
    
    if st.session_state is None:
        st.session_state = {'client_name': "Bambang",
                            'tenure_months': 24,
                            'location': 'Bandung',
                            'device_class': 'Low End',
                            'games_product': 'No',
                            'music_product': 'Yes',
                            'education_product': 'Yes',
                            'video_product': 'Yes',
                            'call_center': 'No',
                            'use_myapp': 'Yes',
                            'payment_method': 'Pulsa',
                            'monthly_purchase_thou_idr_': 70.0,
                            'cltv_predicted_thou_idr_': 3800.0,}
    
    # Initialize session state
    if 'form_content' not in st.session_state:
        
        st.session_state = {'client_name': "Bambang", 
                            'tenure_months': 24,
                            'location': 'Bandung',
                            'device_class': 'Low End',
                            'games_product': 'No',
                            'music_product': 'Yes',
                            'education_product': 'Yes',
                            'video_product': 'Yes',
                            'call_center': 'No',
                            'use_myapp': 'Yes',
                            'payment_method': 'Pulsa',
                            'monthly_purchase_thou_idr_': 70.0,
                            'cltv_predicted_thou_idr_': 3800.0,}
        
        st.session_state['form_content'] = {'client_name': "Bambang",
                                            'tenure_months': 24,
                                            'location': 'Bandung',
                                            'device_class': 'Low End',
                                            'games_product': 'No',
                                            'music_product': 'Yes',
                                            'education_product': 'Yes',
                                            'video_product': 'Yes',
                                            'call_center': 'No',
                                            'use_myapp': 'Yes',
                                            'payment_method': 'Pulsa',
                                            'monthly_purchase_thou_idr_': 70.0,
                                            'cltv_predicted_thou_idr_': 3800.0,}
    
    st.write("Not sure how to? Try our default clients!")
    st.subheader("Meet Supriyanto and Aisyah!")
    
    col_sampleA, col_sampleB = st.columns(2)

    with col_sampleA:
        lottie_url = "https://lottie.host/0db51d3e-e84e-4e5a-8b1e-f73a89a77f65/i1GvROt5y3.json"
        lottie_animation = load_lottie_url(lottie_url)
        st_lottie(lottie_animation, speed=1, width=350, height=350)
        st.markdown(
            "<div style='text-align: center'>Supriyanto</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center'>A 58-year-old technician, from Jakarta, binging Netflix and chill</div><br>", unsafe_allow_html=True)
        
        if st.button("Choose me!"):
            sampleA = {'client_name': 'Supriyanto',
                       'tenure_months': 2,
                       'location': 'Jakarta',
                       'device_class': 'Mid End',
                       'games_product': 'Yes',
                       'music_product': 'Yes',
                       'education_product': 'Yes',
                       'video_product': 'Yes',
                       'call_center': 'No',
                       'use_myapp': 'No',
                       'payment_method': 'Pulsa',
                       'monthly_purchase_thou_idr_': 45.0,
                       'cltv_predicted_thou_idr_': 4210.7,}
            
            # set_form_content(sampleA)
            # st.session_state["client_name"] = sampleA["client_name"]
            st.session_state.get('client_name', sampleA["client_name"])
            update_form_values(sampleA)

    with col_sampleB:
        lottie_url = "https://lottie.host/067bfd39-6ab6-484b-abd1-37451c842fd3/4OhK1ZCsaG.json"
        lottie_animation = load_lottie_url(lottie_url)
        st_lottie(lottie_animation, speed=1, width=350, height=350)
        st.markdown(
            "<div style='text-align: center'>Aisyah</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align: center'>A 38-year-old manager, from Jakarta, all-time professional gamer</div><br>", unsafe_allow_html=True)
        
        if st.button("Pick me!"):
            sampleB = {'client_name': 'Aisyah',
                       'tenure_months': 66,
                       'location': 'Jakarta',
                       'device_class': 'High End',
                       'games_product': 'Yes',
                       'music_product': 'No',
                       'education_product': 'Yes',
                       'video_product': 'Yes',
                       'call_center': 'Yes',
                       'use_myapp': 'Yes',
                       'payment_method': 'Debit',
                       'monthly_purchase_thou_idr_': 137.345,
                       'cltv_predicted_thou_idr_': 6626.1,}
            
            # set_form_content(sampleB)
            # st.session_state["client_name"] = sampleB["client_name"]
            st.session_state.get('client_name', sampleB["client_name"])
            update_form_values(sampleB)
    
    st.divider()        
    
    client_name = st.text_input("Enter your name", st.session_state["client_name"])
    st.session_state["client_name"] = client_name
    
    with st.form('user_inputs'):

        st.header("Chapter 1: The Mysterious Client")
        st.write("Unveiling the Persona")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tenure_months = session_slider_int("How deep are the roots? How many months has the customer stayed? (0-72)", 0, 72, key='tenure_months', default_value=st.session_state.get('tenure_months', 24)) # st.session_state['form_content']['tenure_months']
        
        with col2:
            location = session_selectbox("Where is the spotlight? Which city does the customer reside in?", ["Jakarta", "Bandung"], key='location', default_value=st.session_state.get('location', "Bandung")) # st.session_state['form_content']['location']
            device_class = session_selectbox("What's their device flavour?", ["High End", "Mid End", "Low End"], key='device_class', default_value=st.session_state.get('device_class', "Low End")) # st.session_state['form_content']['device_class']
    
        st.header("Chapter 2: User Vibes: Gamers, Streamers, or Learners?")
        st.write("Unravelling the Lifestyle")
        
        col3, col4 = st.columns(2)
        
        with col3:
            games_product = session_radio("Chasing high scores? Does the customer use the internet service for games?", ["No", "Yes", "No internet service"], key='games_product', default_value=st.session_state.get('games_product', "No")) # st.session_state['form_content']['games_product']
            music_product = session_radio("Is our network a concert hall? Does the customer stream music?", ["No", "Yes", "No internet service"], key='music_product', default_value=st.session_state.get('music_product', "Yes")) # st.session_state['form_content']['music_product']
            education_product = session_radio("An E-Learner or traditional? Does the customer use our educational services?", ["No", "Yes", "No internet service"], key='education_product', default_value=st.session_state.get('education_product', "Yes")) # st.session_state['form_content']['education_product']
            
        with col4:
            video_product = session_radio("Netflix and Chill? Does the customer stream videos?", ["No", "Yes", "No internet service"], key='video_product', default_value=st.session_state.get('video_product', "Yes")) # st.session_state['form_content']['video_product']
            call_center = session_radio("Have the customer sought our voice? Ever called our help center?", ["No", "Yes"], key='call_center', default_value=st.session_state.get('call_center', "No")) # st.session_state['form_content']['call_center']
            use_myapp = session_radio("Is the customer in the MyApp loop? Utilizing MyApp services?", ["No", "Yes", "No internet service"], key='use_myapp', default_value=st.session_state.get('use_myapp', "Yes")) # st.session_state['form_content']['use_myapp']
    
        st.header("Chapter 3: Secrets of the Wallet")
        st.write("Unlocking Financial Chronicles")
        
        col5, col6 = st.columns(2)
        
        with col5:
            payment_method = session_selectbox(
            "How does our customer square up? Pulsa, Digital Wallet, Debit, or Credit?", 
            ["Pulsa", "Digital Wallet", "Debit", "Credit"], 
            key='payment_method', 
            default_value=st.session_state.get('payment_method', "Pulsa") # st.session_state['form_content']['payment_method']
            )
            
            monthly_purchase = session_number_input("How Big is Their Telecom Appetite? Monthly Spend in Thousands of IDR? (20-160)", key='monthly_purchase_thou_idr_', default_value=st.session_state.get('monthly_purchase_thou_idr_', 70.0)) # st.session_state['form_content']['monthly_purchase']
        
        with col6:
            cltv = session_number_input("What's Their Long-Term Value to Us? CLTV in Thousands of IDR? (2500-8500)", key='cltv_predicted_thou_idr_', default_value=st.session_state.get('cltv_predicted_thou_idr_ltv', 3800.0)) # st.session_state['form_content']['cltv']
    
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write(f"Dignosing {client_name}'s churn outcome...") #st.session_state['client_name']
            progress_bar = st.progress(0)
            
            for perc_completed in range(100):
                time.sleep(0.01)
                progress_bar.progress(perc_completed+1)
            
            inputs = {
                'Tenure Months': tenure_months,
                'Location': location,
                'Device Class': device_class,
                'Games Product': games_product,
                'Music Product': music_product,
                'Education Product': education_product,
                'Video Product': video_product,
                'Call Center': call_center,
                'Use MyApp': use_myapp,
                'Payment Method': payment_method,
                'Monthly Purchase (Thou. IDR)': monthly_purchase,
                'CLTV (Predicted Thou. IDR)': cltv,
            }
            prediction = make_prediction(inputs)
            if prediction == 1:
                st.error("Our analysis suggests the client is very likely to churn.")
            else:
                st.success("Our analysis suggests the client is unlikely to churn.")
                
            del(
                col_sampleA,
                col_sampleB,
                client_name,
                col1,
                col2,
                col3,
                col4,
                col5,
                col6,
                submitted,
                inputs,
                prediction
            )
            gc.collect()
            
if __name__ == "__main__":
    main()