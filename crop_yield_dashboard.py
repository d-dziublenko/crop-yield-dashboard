import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import shap
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Crop Yield Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.feature_names = None
    st.session_state.data = None  # Add data to session state


def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic crop yield data for demonstration
    Real implementation would load actual agricultural data
    """
    np.random.seed(42)

    # Feature generation with realistic ranges
    data = pd.DataFrame(
        {
            "temperature": np.random.normal(25, 5, n_samples),  # Celsius
            "rainfall": np.random.exponential(50, n_samples),  # mm per month
            "humidity": np.random.normal(65, 15, n_samples),  # percentage
            "soil_ph": np.random.normal(6.5, 0.8, n_samples),  # pH scale
            "nitrogen": np.random.normal(40, 10, n_samples),  # kg/ha
            "phosphorus": np.random.normal(30, 8, n_samples),  # kg/ha
            "potassium": np.random.normal(35, 9, n_samples),  # kg/ha
            "crop_type": np.random.choice(
                ["wheat", "corn", "rice", "soybeans"], n_samples
            ),
            "field_size": np.random.uniform(1, 50, n_samples),  # hectares
            "irrigation": np.random.choice([0, 1], n_samples),  # binary
            "pesticide_use": np.random.uniform(0, 5, n_samples),  # kg/ha
            "latitude": np.random.uniform(20, 45, n_samples),
            "longitude": np.random.uniform(-120, -70, n_samples),
        }
    )

    # Clip values to realistic ranges
    data["temperature"] = np.clip(data["temperature"], 10, 40)
    data["rainfall"] = np.clip(data["rainfall"], 0, 300)
    data["humidity"] = np.clip(data["humidity"], 20, 95)
    data["soil_ph"] = np.clip(data["soil_ph"], 4.5, 8.5)
    data["nitrogen"] = np.clip(data["nitrogen"], 10, 80)
    data["phosphorus"] = np.clip(data["phosphorus"], 10, 60)
    data["potassium"] = np.clip(data["potassium"], 10, 70)

    # Calculate yield based on features (simplified model)
    yield_base = 3000  # kg/ha base yield

    # Temperature effect (optimal around 25°C)
    temp_effect = -20 * (data["temperature"] - 25) ** 2

    # Rainfall effect (optimal around 100mm)
    rain_effect = -0.5 * (data["rainfall"] - 100) ** 2

    # Nutrient effects
    nutrient_effect = (
        data["nitrogen"] * 5 + data["phosphorus"] * 4 + data["potassium"] * 3
    )

    # pH effect (optimal around 6.5)
    ph_effect = -500 * (data["soil_ph"] - 6.5) ** 2

    # Crop type multipliers
    crop_multipliers = {"wheat": 1.0, "corn": 1.2, "rice": 0.9, "soybeans": 0.8}
    crop_effect = data["crop_type"].map(crop_multipliers)

    # Calculate yield with some noise
    data["yield"] = (
        yield_base + temp_effect + rain_effect + nutrient_effect + ph_effect
    ) * crop_effect
    data["yield"] *= 1 + data["irrigation"] * 0.2  # Irrigation bonus
    data["yield"] *= 1 - data["pesticide_use"] * 0.02  # Pesticide diminishing returns
    data["yield"] += np.random.normal(0, 200, n_samples)  # Random variation
    data["yield"] = np.clip(data["yield"], 500, 8000)  # Realistic yield range

    return data


def train_model(data):
    """Train XGBoost model on the crop yield data"""
    # First, let's check what columns we actually have
    st.write("Data columns found:", data.columns.tolist())

    # Identify categorical and numerical columns dynamically
    categorical_features = []
    numerical_features = []

    # Check for expected columns and handle variations
    for col in data.columns:
        if col.lower() in ["yield", "latitude", "longitude"]:
            continue  # Skip target and location columns
        elif data[col].dtype == "object" or col.lower() in [
            "crop_type",
            "crop",
            "variety",
        ]:
            categorical_features.append(col)
        elif data[col].dtype in ["int64", "float64"]:
            numerical_features.append(col)

    st.write(f"Categorical features detected: {categorical_features}")
    st.write(f"Numerical features detected: {numerical_features}")

    # Check if we have a yield column
    yield_column = None
    for col in data.columns:
        if "yield" in col.lower():
            yield_column = col
            break

    if yield_column is None:
        st.error(
            "No 'yield' column found in the data. Please ensure your data has a column containing yield values."
        )
        return None, None, None, None, None, None, None

    # One-hot encode categorical variables if any exist
    if categorical_features:
        data_encoded = pd.get_dummies(
            data, columns=categorical_features, prefix_sep="_"
        )
    else:
        data_encoded = data.copy()

    # Separate features and target
    feature_cols = [
        col
        for col in data_encoded.columns
        if col not in [yield_column, "latitude", "longitude"]
    ]
    X = data_encoded[feature_cols]
    y = data_encoded[yield_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, scaler, X.columns.tolist(), mse, r2, X_train_scaled, X_test_scaled


def create_shap_plots(model, X_scaled, feature_names):
    """Create SHAP interpretation plots"""
    # Calculate SHAP values
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)

    return explainer, shap_values


# Main app
st.markdown(
    '<h1 class="main-header">🌾 Crop Yield Dashboard</h1>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("Dashboard Controls")

    # Data source selection
    data_source = st.selectbox(
        "Select Data Source",
        ["Synthetic Data (Demo)", "Upload CSV", "Connect to Database"],
    )

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data  # Store in session state
                st.success("Data uploaded successfully!")

                # Display data info to help users understand their data
                with st.expander("📊 Data Preview and Information"):
                    st.write("**First 5 rows of your data:**")
                    st.dataframe(data.head())

                    st.write("**Column Information:**")
                    col_info = pd.DataFrame(
                        {
                            "Column": data.columns,
                            "Type": data.dtypes.astype(str),
                            "Non-Null Count": data.count(),
                            "Unique Values": data.nunique(),
                        }
                    )
                    st.dataframe(col_info)

                    # Check for required columns
                    st.write("**Data Validation:**")
                    has_yield = any("yield" in col.lower() for col in data.columns)
                    if has_yield:
                        st.success("✅ Yield column detected")
                    else:
                        st.error(
                            "❌ No yield column found. Please ensure your data has a column with 'yield' in the name."
                        )

                    # Suggest column mapping if needed
                    if not has_yield:
                        st.info(
                            "💡 Tip: Your yield column should be named something like 'yield', 'crop_yield', or 'yield_per_hectare'"
                        )

            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                data = generate_synthetic_data(1000)
                st.session_state.data = data  # Store in session state
        else:
            # If no file uploaded but data exists in session, use it
            if st.session_state.data is None:
                data = generate_synthetic_data(1000)
                st.session_state.data = data
    else:
        data = generate_synthetic_data(1000)
        st.session_state.data = data  # Store in session state

    st.divider()

    # Model controls
    if st.button("Train Model", type="primary"):
        if st.session_state.data is not None:
            with st.spinner("Training model..."):
                (
                    model,
                    scaler,
                    feature_names,
                    mse,
                    r2,
                    X_train_scaled,
                    X_test_scaled,
                ) = train_model(st.session_state.data)
                if model is not None:  # Check if training was successful
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.feature_names = feature_names
                    st.session_state.model_trained = True
                    st.session_state.X_train_scaled = X_train_scaled
                    st.session_state.X_test_scaled = X_test_scaled
                    st.session_state.mse = mse
                    st.session_state.r2 = r2
                    st.success("Model trained successfully!")
        else:
            st.error("No data available. Please load data first.")

# Main content area
if st.session_state.model_trained and st.session_state.data is not None:
    data = st.session_state.data  # Get data from session state

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model R² Score", f"{st.session_state.r2:.3f}")
    with col2:
        st.metric("RMSE (kg/ha)", f"{np.sqrt(st.session_state.mse):.0f}")
    with col3:
        st.metric("Total Fields", len(data))
    with col4:
        # Find the yield column dynamically
        yield_col = None
        for col in data.columns:
            if "yield" in col.lower():
                yield_col = col
                break
        if yield_col:
            st.metric("Avg Yield (kg/ha)", f"{data[yield_col].mean():.0f}")
        else:
            st.metric("Avg Yield (kg/ha)", "N/A")

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Overview",
            "🔮 Predictions",
            "📈 Model Insights",
            "🗺️ Geographic View",
            "📋 Data Analysis",
        ]
    )

    with tab1:
        st.header("Crop Yield Overview")

        # Find categorical and numerical columns dynamically
        categorical_cols = [col for col in data.columns if data[col].dtype == "object"]
        numerical_cols = [
            col for col in data.columns if data[col].dtype in ["int64", "float64"]
        ]

        # Find yield column
        yield_col = None
        for col in data.columns:
            if "yield" in col.lower():
                yield_col = col
                break

        # Yield distribution by crop type (if categorical columns exist)
        col1, col2 = st.columns(2)

        with col1:
            if categorical_cols and yield_col:
                # Use the first categorical column as proxy for crop type
                cat_col = categorical_cols[0]
                fig_box = px.box(
                    data,
                    x=cat_col,
                    y=yield_col,
                    title=f"Yield Distribution by {cat_col.title()}",
                    labels={yield_col: "Yield (kg/ha)", cat_col: cat_col.title()},
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No categorical columns found for grouping yield data")

        with col2:
            # Correlation heatmap with available numeric columns
            if len(numerical_cols) > 1:
                corr_matrix = data[numerical_cols].corr()

                fig_heatmap = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu",
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Not enough numerical columns for correlation analysis")

        # Time series simulation (would be real dates in production)
        st.subheader("Yield Trends Over Time")
        if categorical_cols and yield_col:
            # Group by first categorical column and show trends
            unique_categories = data[categorical_cols[0]].unique()[
                :4
            ]  # Limit to 4 categories
            dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="M")

            trend_data = {"date": dates}
            for cat in unique_categories:
                cat_yield = data[data[categorical_cols[0]] == cat][yield_col].mean()
                if pd.notna(cat_yield):
                    trend_data[str(cat)] = (
                        np.random.normal(
                            cat_yield, cat_yield * 0.1, len(dates)
                        ).cumsum()
                        / len(dates)
                        + cat_yield * 0.9
                    )

            monthly_yields = pd.DataFrame(trend_data)

            fig_timeline = px.line(
                monthly_yields.melt(
                    id_vars="date", var_name="category", value_name="yield"
                ),
                x="date",
                y="yield",
                color="category",
                title=f"Monthly Average Yield by {categorical_cols[0].title()}",
                labels={"yield": "Yield (kg/ha)", "date": "Date"},
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info(
                "Time series visualization requires categorical data and yield values"
            )

    with tab2:
        st.header("Yield Prediction Tool")

        # Create input form
        col1, col2, col3 = st.columns(3)

        with col1:
            crop_type = st.selectbox("Crop Type", ["wheat", "corn", "rice", "soybeans"])
            temperature = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)
            rainfall = st.slider("Rainfall (mm/month)", 0.0, 300.0, 100.0)
            humidity = st.slider("Humidity (%)", 20.0, 95.0, 65.0)

        with col2:
            soil_ph = st.slider("Soil pH", 4.5, 8.5, 6.5)
            nitrogen = st.slider("Nitrogen (kg/ha)", 10.0, 80.0, 40.0)
            phosphorus = st.slider("Phosphorus (kg/ha)", 10.0, 60.0, 30.0)
            potassium = st.slider("Potassium (kg/ha)", 10.0, 70.0, 35.0)

        with col3:
            field_size = st.number_input("Field Size (hectares)", 1.0, 50.0, 10.0)
            irrigation = st.selectbox("Irrigation", ["No", "Yes"])
            pesticide_use = st.slider("Pesticide Use (kg/ha)", 0.0, 5.0, 1.0)

        if st.button("Predict Yield", type="primary"):
            # Create a dictionary for input data based on available features
            input_dict = {}

            # Map the input controls to potential column names in the data
            feature_mapping = {
                "temperature": temperature,
                "temp": temperature,
                "rainfall": rainfall,
                "rain": rainfall,
                "precipitation": rainfall,
                "humidity": humidity,
                "soil_ph": soil_ph,
                "ph": soil_ph,
                "nitrogen": nitrogen,
                "n": nitrogen,
                "phosphorus": phosphorus,
                "p": phosphorus,
                "potassium": potassium,
                "k": potassium,
                "field_size": field_size,
                "size": field_size,
                "irrigation": 1 if irrigation == "Yes" else 0,
                "pesticide_use": pesticide_use,
                "pesticide": pesticide_use,
            }

            # Add crop type with various possible column names
            crop_mapping = {
                "crop_type": crop_type,
                "crop": crop_type,
                "variety": crop_type,
                "type": crop_type,
            }

            # Build input data based on actual feature names
            for feature_name in st.session_state.feature_names:
                # Check if it's a numerical feature
                for key, value in feature_mapping.items():
                    if key in feature_name.lower():
                        input_dict[feature_name] = value
                        break

                # Check if it's a categorical feature (one-hot encoded)
                for key, value in crop_mapping.items():
                    if (
                        feature_name.startswith(f"{key}_")
                        and value.lower() in feature_name.lower()
                    ):
                        input_dict[feature_name] = 1
                    elif feature_name.startswith(f"{key}_"):
                        input_dict[feature_name] = 0

            # Create DataFrame from input
            input_data = pd.DataFrame([input_dict])

            # Ensure all columns are present
            for col in st.session_state.feature_names:
                if col not in input_data.columns:
                    input_data[col] = 0

            input_encoded = input_data[st.session_state.feature_names]

            # Scale and predict
            input_scaled = st.session_state.scaler.transform(input_encoded)
            prediction = st.session_state.model.predict(input_scaled)[0]

            # Display prediction
            st.success(f"### Predicted Yield: {prediction:.0f} kg/ha")

            # Show feature importance for this prediction
            explainer = shap.Explainer(
                st.session_state.model, st.session_state.X_train_scaled
            )
            shap_values = explainer(input_scaled)

            # Create waterfall plot
            st.subheader("Prediction Explanation")
            feature_names_display = [
                name.replace("crop_type_", "")
                for name in st.session_state.feature_names
            ]

            # Get SHAP values and create a dataframe for plotting
            shap_df = pd.DataFrame(
                {
                    "Feature": feature_names_display,
                    "SHAP Value": shap_values.values[0],
                    "Feature Value": input_encoded.values[0],
                }
            )
            shap_df = shap_df[shap_df["SHAP Value"] != 0].sort_values(
                "SHAP Value", key=abs, ascending=False
            )

            fig_waterfall = go.Figure(
                go.Waterfall(
                    x=shap_df["Feature"][:10],
                    y=shap_df["SHAP Value"][:10],
                    text=[f"{val:.0f}" for val in shap_df["SHAP Value"][:10]],
                    textposition="outside",
                )
            )
            fig_waterfall.update_layout(
                title="Top 10 Features Impact on Prediction",
                yaxis_title="Impact on Yield (kg/ha)",
                xaxis_title="Features",
                showlegend=False,
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

    with tab3:
        st.header("Model Insights")

        # Feature importance
        importance_df = pd.DataFrame(
            {
                "feature": st.session_state.feature_names,
                "importance": st.session_state.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        fig_importance = px.bar(
            importance_df.head(15),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Feature Importances",
            labels={"importance": "Importance Score", "feature": "Feature"},
        )
        st.plotly_chart(fig_importance, use_container_width=True)

        # SHAP summary plot
        st.subheader("SHAP Analysis")

        # Calculate SHAP values for a sample of the test set
        sample_size = min(100, len(st.session_state.X_test_scaled))
        explainer = shap.Explainer(
            st.session_state.model, st.session_state.X_train_scaled
        )
        shap_values = explainer(st.session_state.X_test_scaled[:sample_size])

        # Create beeswarm plot
        shap_df = pd.DataFrame(
            shap_values.values, columns=st.session_state.feature_names
        )

        # Get mean absolute SHAP values for each feature
        mean_shap = shap_df.abs().mean().sort_values(ascending=False).head(10)

        fig_shap = px.bar(
            x=mean_shap.values,
            y=mean_shap.index,
            orientation="h",
            title="Mean Absolute SHAP Values (Top 10 Features)",
            labels={"x": "Mean |SHAP value|", "y": "Feature"},
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # Partial dependence plots
        st.subheader("Feature Effects on Yield")

        # Get available numerical features for partial dependence
        available_features = [
            f
            for f in st.session_state.feature_names
            if any(
                keyword in f.lower()
                for keyword in [
                    "temperature",
                    "temp",
                    "rainfall",
                    "rain",
                    "ph",
                    "nitrogen",
                    "n",
                ]
            )
        ]

        if available_features:
            feature_to_plot = st.selectbox(
                "Select feature for partial dependence", available_features
            )

            # Check if the selected feature exists in the original data
            original_feature = None
            for col in data.columns:
                if (
                    col.lower() in feature_to_plot.lower()
                    or feature_to_plot.lower() in col.lower()
                ):
                    original_feature = col
                    break

            if original_feature and original_feature in data.columns:
                # Create partial dependence data
                feature_values = np.linspace(
                    data[original_feature].min(), data[original_feature].max(), 50
                )

                # Calculate partial dependence
                predictions = []

                for val in feature_values:
                    # Create a copy of the data for prediction
                    temp_data = data.sample(
                        n=min(50, len(data)), random_state=42
                    ).copy()
                    temp_data[original_feature] = val

                    # Process the data similar to training
                    categorical_features = [
                        col
                        for col in temp_data.columns
                        if temp_data[col].dtype == "object"
                    ]
                    if categorical_features:
                        temp_encoded = pd.get_dummies(
                            temp_data, columns=categorical_features, prefix_sep="_"
                        )
                    else:
                        temp_encoded = temp_data.copy()

                    # Remove target and location columns
                    feature_cols = [
                        col
                        for col in temp_encoded.columns
                        if not any(
                            x in col.lower() for x in ["yield", "latitude", "longitude"]
                        )
                    ]
                    temp_encoded = temp_encoded[feature_cols]

                    # Ensure all columns match the training features
                    for col in st.session_state.feature_names:
                        if col not in temp_encoded.columns:
                            temp_encoded[col] = 0

                    temp_encoded = temp_encoded[st.session_state.feature_names]
                    temp_scaled = st.session_state.scaler.transform(temp_encoded)
                    pred = st.session_state.model.predict(temp_scaled).mean()
                    predictions.append(pred)

                fig_pd = px.line(
                    x=feature_values,
                    y=predictions,
                    title=f"Partial Dependence: {original_feature.title()}",
                    labels={
                        "x": original_feature.title(),
                        "y": "Predicted Yield (kg/ha)",
                    },
                )
                st.plotly_chart(fig_pd, use_container_width=True)
            else:
                st.info(
                    "Please select a feature that exists in the original data for partial dependence plotting."
                )
        else:
            st.info("No suitable features found for partial dependence analysis.")

    with tab4:
        st.header("Geographic Distribution")

        # Check if location data exists
        has_location = "latitude" in data.columns and "longitude" in data.columns

        if has_location:
            # Create map
            m = folium.Map(
                location=[data["latitude"].mean(), data["longitude"].mean()],
                zoom_start=5,
            )

            # Find yield column
            yield_col = None
            for col in data.columns:
                if "yield" in col.lower():
                    yield_col = col
                    break

            # Find categorical column for labeling
            categorical_cols = [
                col for col in data.columns if data[col].dtype == "object"
            ]
            label_col = categorical_cols[0] if categorical_cols else None

            # Add yield data to map
            if yield_col:
                yield_median = data[yield_col].median()
                for idx, row in data.iterrows():
                    if idx < 200:  # Limit points for performance
                        color = "green" if row[yield_col] > yield_median else "orange"
                        popup_text = f"Yield: {row[yield_col]:.0f} kg/ha"
                        if label_col:
                            popup_text = (
                                f"{label_col.title()}: {row[label_col]}<br>"
                                + popup_text
                            )

                        folium.CircleMarker(
                            location=[row["latitude"], row["longitude"]],
                            radius=5,
                            popup=popup_text,
                            color=color,
                            fill=True,
                            fillColor=color,
                        ).add_to(m)

            # Display map
            st_folium(m, width=700, height=500)

            # Regional statistics
            st.subheader("Regional Yield Statistics")

            if yield_col:
                # Create regions based on latitude
                data["region"] = pd.cut(
                    data["latitude"], bins=3, labels=["South", "Central", "North"]
                )

                if categorical_cols:
                    regional_stats = (
                        data.groupby(["region", categorical_cols[0]])[yield_col]
                        .agg(["mean", "std", "count"])
                        .reset_index()
                    )

                    fig_regional = px.bar(
                        regional_stats,
                        x=categorical_cols[0],
                        y="mean",
                        color="region",
                        error_y="std",
                        title=f"Average Yield by Region and {categorical_cols[0].title()}",
                        labels={
                            "mean": "Average Yield (kg/ha)",
                            categorical_cols[0]: categorical_cols[0].title(),
                        },
                        barmode="group",
                    )
                else:
                    regional_stats = (
                        data.groupby(["region"])[yield_col]
                        .agg(["mean", "std", "count"])
                        .reset_index()
                    )

                    fig_regional = px.bar(
                        regional_stats,
                        x="region",
                        y="mean",
                        error_y="std",
                        title="Average Yield by Region",
                        labels={"mean": "Average Yield (kg/ha)", "region": "Region"},
                    )

                st.plotly_chart(fig_regional, use_container_width=True)
            else:
                st.warning("No yield data found for regional analysis")
        else:
            st.info(
                "Geographic visualization requires 'latitude' and 'longitude' columns in your data."
            )
            st.write("Your current columns are:", list(data.columns))

    with tab5:
        st.header("Data Analysis")

        # Display data statistics
        st.subheader("Dataset Overview")
        st.write(f"Total Records: {len(data)}")
        st.write(f"Features: {len(data.columns) - 1}")
        st.write(f"Target Variable: Yield (kg/ha)")

        # Show data sample
        st.subheader("Data Sample")
        st.dataframe(data.head(10))

        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(data.describe())

        # Missing values check
        st.subheader("Data Quality")
        missing_values = data.isnull().sum()
        if missing_values.sum() == 0:
            st.success("No missing values found in the dataset!")
        else:
            st.warning("Missing values detected:")
            st.dataframe(missing_values[missing_values > 0])

        # Distribution plots
        st.subheader("Feature Distributions")

        feature_to_plot = st.selectbox(
            "Select feature to visualize",
            data.select_dtypes(include=[np.number]).columns.tolist(),
        )

        fig_dist = px.histogram(
            data,
            x=feature_to_plot,
            nbins=30,
            title=f"Distribution of {feature_to_plot.title()}",
            labels={feature_to_plot: feature_to_plot.title(), "count": "Frequency"},
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Export functionality
        st.subheader("Export Options")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Download Predictions CSV"):
                # Create predictions for all data
                data_for_pred = data.drop(["yield", "latitude", "longitude"], axis=1)
                data_encoded = pd.get_dummies(
                    data_for_pred, columns=["crop_type"], prefix_sep="_"
                )

                for col in st.session_state.feature_names:
                    if col not in data_encoded.columns:
                        data_encoded[col] = 0

                data_encoded = data_encoded[st.session_state.feature_names]
                data_scaled = st.session_state.scaler.transform(data_encoded)
                predictions = st.session_state.model.predict(data_scaled)

                export_df = data.copy()
                export_df["predicted_yield"] = predictions
                export_df["yield_difference"] = (
                    export_df["predicted_yield"] - export_df["yield"]
                )

                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="crop_yield_predictions.csv",
                    mime="text/csv",
                )

        with col2:
            if st.button("Save Model"):
                # In production, you would save to a proper location
                st.info(
                    "Model saved successfully! (In production, this would save to your model registry)"
                )

else:
    # Landing page when model is not trained
    st.info("👈 Please train the model using the sidebar controls to begin analysis")

    # Show demo information
    st.header("Welcome to the Crop Yield Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        ### 🌾 Features
        - Machine learning-based yield prediction
        - Interactive visualizations
        - Geographic analysis
        - Model interpretability with SHAP
        """
        )

    with col2:
        st.markdown(
            """
        ### 📊 Data Sources
        - Synthetic demo data
        - CSV file upload
        - Database connectivity (coming soon)
        - Real-time weather APIs (coming soon)
        """
        )

    with col3:
        st.markdown(
            """
        ### 🎯 Use Cases
        - Crop planning optimization
        - Risk assessment
        - Resource allocation
        - Yield forecasting
        """
        )

    # Display sample data structure
    st.subheader("Expected Data Format")
    sample_data = pd.DataFrame(
        {
            "temperature": [25.0],
            "rainfall": [100.0],
            "humidity": [65.0],
            "soil_ph": [6.5],
            "nitrogen": [40.0],
            "phosphorus": [30.0],
            "potassium": [35.0],
            "crop_type": ["wheat"],
            "field_size": [10.0],
            "irrigation": [1],
            "pesticide_use": [2.0],
            "yield": [3500.0],
        }
    )
    st.dataframe(sample_data)

# Footer
st.markdown("---")
st.markdown(
    "Built with Streamlit, XGBoost, and SHAP | Agricultural Intelligence Platform"
)
