import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

# Suppress inconsistent version warnings from sklearn
warnings.filterwarnings("ignore", category=UserWarning)

# Set page configuration
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    
    /* Main title styling with gradient */
    .main-title {
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .subtitle {
        color: #718096;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Glassmorphism containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
    }
    
    /* Prediction output card styling */
    .prediction-card {
        background: linear-gradient(135deg, rgba(77, 150, 255, 0.15) 0%, rgba(255, 107, 107, 0.15) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 10px 40px 0 rgba(77, 150, 255, 0.2);
        animation: fadeIn 0.8s ease-out;
    }
    
    .price-display {
        font-size: 3.5rem;
        font-weight: 800;
        color: #4D96FF;
        text-shadow: 0 0 20px rgba(77, 150, 255, 0.4);
        margin: 15px 0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom button styling */
    .stButton>button {
        background: linear-gradient(135deg, #4D96FF 0%, #00C9A7 100%) !important;
        color: white !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        border: none !important;
        padding: 12px 30px !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(77, 150, 255, 0.3) !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(77, 150, 255, 0.5) !important;
        background: linear-gradient(135deg, #00C9A7 0%, #4D96FF 100%) !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Find model and scaler
# Define possible locations for the model and scaler
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
possible_locations = [
    # Relative to root directory (when streamlit run webapp/app.py)
    ("ml model/model.pkl", "ml model/scaler.pkl"),
    ("ml_model/model.pkl", "ml_model/scaler.pkl"),
    # Relative to app.py directory
    (os.path.join(base_dir, "ml model", "model.pkl"), os.path.join(base_dir, "ml model", "scaler.pkl")),
    (os.path.join(base_dir, "ml_model", "model.pkl"), os.path.join(base_dir, "ml_model", "scaler.pkl")),
    # Fallback to current directory check
    ("model.pkl", "scaler.pkl")
]

model = None
scaler = None
loaded_path = ""

for model_p, scaler_p in possible_locations:
    if os.path.exists(model_p) and os.path.exists(scaler_p):
        try:
            model = joblib.load(model_p)
            scaler = joblib.load(scaler_p)
            loaded_path = model_p
            break
        except Exception as e:
            continue

# App header
st.markdown("<div class='main-title'>California House Price Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict the median house price in a California district based on 1990 census features.</div>", unsafe_allow_html=True)

if model is None or scaler is None:
    st.error("❌ Model or Scaler could not be found! Please make sure model.pkl and scaler.pkl are placed in the 'ml model' folder.")
    st.info("Currently searched paths:\n" + "\n".join([f"- Model: {m}\n  Scaler: {s}" for m, s in possible_locations]))
else:
    # Sidebar features
    st.sidebar.markdown("### 📊 Features Configuration")
    st.sidebar.write("Adjust the sliders below to modify the input parameters:")

    # Sliders for California housing features
    med_inc = st.sidebar.slider(
        "MedInc (Median Income)",
        min_value=0.5,
        max_value=15.0,
        value=3.87,
        step=0.1,
        help="Median income in block group (in $10,000s, e.g. 5.0 = $50,000)"
    )

    house_age = st.sidebar.slider(
        "HouseAge (Median House Age)",
        min_value=1.0,
        max_value=52.0,
        value=28.0,
        step=1.0,
        help="Median age of houses in the block group"
    )

    ave_rooms = st.sidebar.slider(
        "AveRooms (Average Rooms)",
        min_value=1.0,
        max_value=15.0,
        value=5.4,
        step=0.1,
        help="Average number of rooms per household"
    )

    ave_bedrms = st.sidebar.slider(
        "AveBedrms (Average Bedrooms)",
        min_value=0.5,
        max_value=5.0,
        value=1.1,
        step=0.1,
        help="Average number of bedrooms per household"
    )

    population = st.sidebar.slider(
        "Population",
        min_value=3.0,
        max_value=10000.0,
        value=1425.0,
        step=10.0,
        help="Total population in the block group"
    )

    ave_occup = st.sidebar.slider(
        "AveOccup (Average Occupants)",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.1,
        help="Average number of household members"
    )

    latitude = st.sidebar.slider(
        "Latitude",
        min_value=32.5,
        max_value=42.5,
        value=35.6,
        step=0.01,
        help="Latitude of the block group"
    )

    longitude = st.sidebar.slider(
        "Longitude",
        min_value=-124.5,
        max_value=-114.3,
        value=-119.5,
        step=0.01,
        help="Longitude of the block group"
    )

    # Creating two columns for main panel
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### 📋 Input Summary")
        st.write("Confirm your selected features:")
        
        # Display inputs in a styled DataFrame/table
        input_data = {
            "Feature": [
                "Median Income ($10k)", 
                "House Age", 
                "Avg Rooms", 
                "Avg Bedrooms", 
                "Population", 
                "Avg Occupants", 
                "Latitude", 
                "Longitude"
            ],
            "Value": [
                f"${med_inc*10:.2f}k / year", 
                f"{int(house_age)} years", 
                f"{ave_rooms:.2f}", 
                f"{ave_bedrms:.2f}", 
                f"{int(population)}", 
                f"{ave_occup:.2f}", 
                f"{latitude:.4f}° N", 
                f"{longitude:.4f}° W"
            ]
        }
        st.table(pd.DataFrame(input_data))

        # Predict Button
        predict_clicked = st.button("🔮 Predict House Price")

    with col2:
        st.markdown("### 📍 Location Map")
        st.write("The house block group is located here:")
        
        # Plot selected location on a map
        map_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
        st.map(map_df, zoom=6, use_container_width=True)

    # Display prediction result
    if predict_clicked:
        # Prepare feature vector matching training order
        feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        features_df = pd.DataFrame([[
            med_inc, house_age, ave_rooms, ave_bedrms, 
            population, ave_occup, latitude, longitude
        ]], columns=feature_names)
        
        try:
            # Scale features
            scaled_features = scaler.transform(features_df)
            
            # Predict
            raw_prediction = model.predict(scaled_features)[0]
            
            # Convert prediction from $100k units to absolute USD
            predicted_price_usd = raw_prediction * 100000
            
            # Display prediction card
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div style="font-size: 1.3rem; font-weight: 600; color: #718096;">PREDICTED HOUSE VALUE</div>
                    <div class="price-display">${predicted_price_usd:,.2f}</div>
                    <div style="font-size: 0.95rem; color: #718096; margin-top: 10px;">
                        Based on the pre-trained Random Forest model.<br>
                        <span style="font-size: 0.8rem; font-style: italic;">Model loaded from: {os.path.basename(os.path.dirname(loaded_path))}/{os.path.basename(loaded_path)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error during prediction: {e}")
