import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from preprocessing_utils import predict_with_confidence_analysis

# Load model and config
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_model.h5')
    with open('model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    return model, config['char_mapping']

st.title("🖋️ Handwritten Character Recognition")
st.write("Upload an image with handwritten text")

# Load model
try:
    model, char_mapping = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Save temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Analyze button
    if st.button('🔍 Analyze Handwriting'):
        with st.spinner('Analyzing...'):
            try:
                result = predict_with_confidence_analysis(
                    "temp_image.jpg", 
                    model, 
                    char_mapping, 
                    show_plots=False
                )
                
                if result:
                    text, predictions, low_conf = result
                    
                    # Results
                    st.success("✅ Analysis Complete!")
                    st.subheader("📝 Recognized Text:")
                    st.code(text, language=None)
                    
                    # Stats
                    col1, col2, col3 = st.columns(3)
                    char_confidences = [p[1] for p in predictions if p[0] != ' ']
                    avg_conf = np.mean(char_confidences) if char_confidences else 0
                    
                    with col1:
                        st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
                    with col2:
                        st.metric("Total Characters", len(char_confidences))
                    with col3:
                        st.metric("Low Confidence", len(low_conf))
                        
                else:
                    st.error("❌ No text detected")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")