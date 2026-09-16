import streamlit as st
import requests
import sys
import os
import time


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from backend.constants import SUB_CATEGORIES, CATEGORIES, SEGMENTS, REGIONS


BASE_DIR = os.path.dirname(__file__)



API_URL = os.getenv(
    "API_URL",
    "https://global-superstore-profit-prediction.onrender.com"
)

APP_URL = f"{API_URL}/predict"
HEALTH_URL = f"{API_URL}/health"


dashboard_img = os.path.join(
    BASE_DIR,
    "Images",
    "dashboard.png"
)

rf_img = os.path.join(
    BASE_DIR,
    "Images",
    "rf_feature_importance.png"
)

shap = os.path.join(
    BASE_DIR,
    "Images",
    "shap_explain.png"
)


st.set_page_config(
    page_title="Global Superstore Analytics",
    layout="wide"
)

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Dataset Overview",
        "EDA & Insights",
        "KPI Dashboard",
        "Profit Prediction",
        "About"
    ]
)

# 1. Home Page 
if menu == "Home":
    st.title("🏪 Global Superstore Analytics")

    st.image(dashboard_img)


    st.markdown("""
    ### Project Highlights

    - 51K+ Orders
    - 140+ Countries
    - Profit Prediction using ML
    - Ensemble Model (RF + GB)
    - R² ≈ 0.77
    - Business Intelligence Dashboard
    """)

# 2. Dataset Overview
elif menu == "Dataset Overview":

    st.header("📊 Dataset Overview")

    st.write("""
    The Global Superstore dataset contains:
    - 51,289 Orders
    - 140+ Countries
    - Sales, Profit, Discount
    - Shipping Cost
    - Category & Sub-Category
    - Region & Market
    """)

    


elif menu == "EDA & Insights":

    st.header("📈 EDA & Insights")

    st.subheader("Feature Importance Analysis")

    st.image(rf_img)

    st.subheader("SHAP Explainability")

    st.image(shap)

    st.info("""
### Key Insights

✅ Sales and discount-related features are the strongest drivers of profit predictions.

✅ Discount strategies can significantly impact profitability, depending on the sales value and product type.

✅ High sales do not always translate into higher profit.

✅ Shipping and product characteristics also contribute to profit outcomes, though with lower influence.

✅ SHAP explanations provide transparency by showing which features increase or decrease the predicted profit for each order.
""")
    

# 4. KPI Dashboard
elif menu == "KPI Dashboard":

    st.header("📋 KPI Dashboard")


    col1, col2, col3 = st.columns(3)

    col1.metric("Revenue", "$12.6M")
    col2.metric("Profit", "$1.47M")
    col3.metric("ROI", "11.6%")

    st.metric("Loss Making Orders", "24.5%")


# 5. Profit Prediction Inputs
elif menu == "Profit Prediction":

    st.header("Profit Prediction")

    

    sales = st.number_input(
    "Sales($)",
    min_value=1.0,
    value=100.0,
    step=1.0
)
    if sales < 15:
        st.warning(
        "Predictions for very small sales values may be less reliable."
    )
    

    discount = st.number_input(
        "Discount",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01
       )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=1,
        step=1
    )

    shipping_cost = st.number_input(
        "Shipping Cost($)",
         min_value=0.0,
         value=0.0,
         step=0.5
    )

    category = st.selectbox(
    "Category",
    list(SUB_CATEGORIES.keys())
)

    sub_category = st.selectbox(
    "Sub-Category",
    SUB_CATEGORIES[category]
    )

    

    segment = st.selectbox(
    "Segment",
    SEGMENTS
    )

    region = st.selectbox(
    "Region",
    REGIONS
    )

    

    order_date = st.date_input(
    "Order Date"
    )

    # Button comes here
    if st.button("Predict Profit"):

        missing = []

        if sales <= 0:
            missing.append("Sales")

        if quantity <= 0:
            missing.append("Quantity")

        if shipping_cost < 0:
            missing.append("Shipping Cost")

        # Discount can be 0, so check differently
        if discount is None:
            missing.append("Discount")

        if missing:
            st.warning(
            "Please fill the following required fields: "
            + ", ".join(missing)
        )

        else:
            payload = {
                "sales": sales,
                "discount": discount,
                "quantity": quantity,
                "shipping_cost": shipping_cost,
                "category": category,
                "segment": segment,
                "sub_category": sub_category,
                "region": region,
                "order_date": str(order_date)
            }

            
            status_box = st.empty()
            backend_ready = False
            MAX_ATTEMPTS = 18

            for attempt in range(1, MAX_ATTEMPTS + 1):
                status_box.info(
                    f"⏳ Waking up backend… attempt {attempt}/{MAX_ATTEMPTS}. "
                    "Render free tier sleeps after inactivity — this may take up to 2 min."
                )
                try:
                    r = requests.get(HEALTH_URL, timeout=10)
                    if r.status_code == 200:
                        backend_ready = True
                        status_box.empty()
                        break
                except Exception as e:
                    status_box.warning(
            f"⏳ Attempt {attempt}/{MAX_ATTEMPTS} — backend not ready yet. "
            "Server may be in sleep mode, retrying..."
        )

                if attempt < MAX_ATTEMPTS:
                  time.sleep(10)

            if not backend_ready:
                st.error("Backend could not be reached after 2 minutes. Please try again.")
                st.stop()

            response = requests.post(APP_URL, json=payload)

            if response.status_code == 200:

                result = response.json()

                pred = result["predicted_profit"]

                if pred > 0:
                    st.success(f"💰 Predicted Profit: ${pred:.2f}")
                else:
                    st.error(f"📉 Predicted Loss: ${abs(pred):.2f}")

                st.info(result["message"])

                # SHAP Explanation
                st.subheader("🔍 Model Explanation")

                st.write("### Positive Drivers")

                shown_messages = set()

                for item in result["shap_explanation"]["positive_drivers"]:
                    feature = item["feature"]

                    if feature == "Discount":
                        message = "Discount"

                    elif feature == "Region: West":
                        message = "West region"

                    elif feature == "Quantity":
                        message = "Order quantity"

                    elif feature == "Shipping Cost Ratio":
                        message = "Sales-to-shipping ratio"

                    else:
                        message = feature

                    if message not in shown_messages:
                        st.success(f"✅ {message} ({item['shap_value']:.2f})")
                        shown_messages.add(message)


                st.write("### Negative Drivers")

                shown_messages = set()

                for item in result["shap_explanation"]["negative_drivers"]:
                    feature = item["feature"]

                    if feature == "Sub-Category: Copiers":
                        message = "Copiers category"

                    elif feature in ["Sales", "Sales (Log Scale)"]:
                        message = "Sales"

                    elif feature == "Shipping Cost":
                        message = "Shipping cost"

                    else:
                        message = feature

                    if message not in shown_messages:
                        st.warning(f"⚠️ {message} ({item['shap_value']:.2f})")
                        shown_messages.add(message)
            else:
                st.error(f"API Error: {response.status_code}")
                st.write(response.text)

# 6. About Project
elif menu == "About":

    st.header("👨‍💻 About Project")

    st.markdown("""### 🌍 Global Superstore Analytics & Profit Prediction

Developed an end-to-end retail analytics, machine learning, and explainable AI solution using the Global Superstore dataset. The project combines data analysis, statistical validation, predictive modeling, model interpretability, and interactive deployment to support business decision-making.

### 📊 Dataset

* 51,000+ Retail Orders
* 140+ Countries
* Multiple Years of Transaction Data
* Sales, Profit, Discount, Shipping, Customer, and Regional Information

### 🔧 Project Components

* Data Cleaning & Preprocessing
* Feature Engineering
* Exploratory Data Analysis (EDA)
* Statistical Hypothesis Testing
* Machine Learning Model Development
* Ensemble Learning (Random Forest + Gradient Boosting)
* SHAP-Based Model Explainability
* FastAPI Backend Development
* Interactive Streamlit Dashboard
* Docker Containerization

### 📈 Hypothesis Testing

* H1: Profit differs across customer segments
* H2: Technology products are more profitable than Furniture
* H3: Discounts negatively impact profit
* H4: Profit differs across global markets
* H5: Same-Day shipping improves profitability

### 🤖 Machine Learning Solution

* Compared multiple regression models
* Built an Ensemble Model using:

  * Random Forest Regressor
  * Gradient Boosting Regressor
* Achieved R² ≈ 0.77 on unseen data

### 🔍 Explainable AI (SHAP)

Implemented SHAP (SHapley Additive Explanations) to make model predictions transparent and interpretable. 
The system identifies the top positive and negative factors influencing each profit prediction, 
helping users understand why an order is expected to be profitable or loss-making.

Examples of explained drivers:

* Discount Impact
* Sales Contribution
* Shipping Cost Effect
* Product Category Influence
* Regional Profitability Patterns

### 💡 Key Business Insights

* High discounts significantly reduce profitability
* Technology is the most profitable category
* APAC generates the highest overall profit
* Approximately 25% of orders are loss-making
* Shipping and discount strategies strongly influence profit outcomes

### 🚀 Deployment

* FastAPI REST API for prediction serving
* Streamlit Web Application for user interaction

### 🛠️ Technologies Used

Python, Pandas, NumPy, Scikit-Learn, SHAP, FastAPI, Streamlit, 
Docker, Matplotlib, Seaborn, Joblib, Pydantic

""")