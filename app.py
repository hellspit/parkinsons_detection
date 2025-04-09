import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import Layer

# -------------------- Custom Attention Layer --------------------
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# -------------------- Load Models --------------------
model_ann = load_model('ann.h5')
model_hybrid = load_model("hybrid.h5", custom_objects={'Attention': Attention})

# -------------------- Load Scalers --------------------
scaler_ann = joblib.load('scaler_ann.pkl')
scaler_hybrid = joblib.load('scaler_hybrid.pkl')

# -------------------- Feature List --------------------
features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE'
]

# -------------------- Streamlit App --------------------
st.title("🧠 Parkinson's Disease Prediction App")

# Input method selection
input_method = st.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

# Input data collection
input_df = None
if input_method == "Manual Entry":
    st.markdown("### Enter feature values:")
    user_input = {feature: st.number_input(f"{feature}", value=0.0) for feature in features}
    input_df = pd.DataFrame([user_input])

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with all required features", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.success("CSV uploaded successfully!")
        st.markdown("### Preview of Uploaded Data")
        st.dataframe(input_df)

# Model selection
model_choice = st.selectbox("Choose a model to use:", ["ANN", "Hybrid (CNN + BiLSTM + Attention)"])

# Predict button
if st.button("Predict"):
    if input_df is not None:
        try:
            # Ensure all required features are present
            missing = [f for f in features if f not in input_df.columns]
            if missing:
                st.error(f"Missing features in input: {missing}")
            else:
                # Apply appropriate scaler
                if model_choice == "ANN":
                    input_scaled = scaler_ann.transform(input_df[features])
                    prediction_probs = model_ann.predict(input_scaled)
                else:
                    input_scaled = scaler_hybrid.transform(input_df[features])
                    input_reshaped = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))
                    prediction_probs = model_hybrid.predict(input_reshaped)

                # Process prediction
                prediction_labels = ["Parkinson's" if p[0] > 0.5 else "Healthy" for p in prediction_probs]
                confidence_scores = [f"{p[0] * 100:.2f}%" for p in prediction_probs]

                # Combine with input data
                result_df = input_df.copy()
                result_df["Prediction"] = prediction_labels
                result_df["Confidence"] = confidence_scores

                # Display results
                st.subheader("🧾 Prediction Results")
                st.dataframe(result_df)

                for i, row in result_df.iterrows():
                    st.success(f"Sample {i + 1}: **{row['Prediction']}** (Confidence: {row['Confidence']})")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
    else:
        st.warning("⚠️ Please provide input data to make predictions.")
