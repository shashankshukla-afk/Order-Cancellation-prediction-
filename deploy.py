import streamlit as st
import joblib
import pandas as pd
import uuid


page = st.sidebar.radio(
    "Select Prediction Mode",[
        "Single Order Prediction",
        "Bulk Prediction"
    ]
)
if page == "Single Order Prediction":
    st.title("Order Cancellation Prediction")
    # 1. Customer Type
    customer_type = st.selectbox(
        "Customer Type",
        ["New", "Returning"],
        key="customer_type_input"
    )

    # 2. Previous Orders
    previous_orders = st.number_input(
        "Previous Orders",
        min_value=0,
        value=0,
        step=1,
        key="previous_orders_input"
    )

    # 3. Previous Cancellations
    previous_cancellations = st.number_input(
        "Previous Cancellations",
        min_value=0,
        value=0,
        step=1,
        key="previous_cancellations_input"
    )

    # 4. Order Value
    order_value = st.number_input(
        "Order Value (₹)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        key="order_value_input"
    )

    # 5. Number of Items
    quantity = st.number_input(
        "Number of Items",
        min_value=1,
        value=1,
        step=1,
        key="quantity_input"
    )

    # 6. Payment Method
    payment_method = st.selectbox(
        "Payment Method",
        ["cash_on_delivery", "card", "paypal", "gift_card", "klarna"],
        key="payment_method_input"
    )

    # 7. Estimated Delivery Days
    days_to_deliver = st.number_input(
        "Estimated Delivery Days",
        min_value=0,
        value=3,
        step=1,
        key="delivery_days_input"
    )

    # 8. Discount Percentage
    discount_pct = st.number_input(
        "Discount Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        key="discount_input"
    )

    # 9. Shipping Cost
    shipping_cost = st.number_input(
        "Shipping Cost (₹)",
        min_value=0.0,
        value=50.0,
        step=10.0,
        key="shipping_cost_input"
    )

    # 10. Product Category
    product_category = st.selectbox(
        "Product Category",
        ["beauty", "electronics", "grocery", "home", "outdoor"],
        key="product_category_input"
    )

    # Load the trained model
    xgb_final = joblib.load("xgboost_undersampled_model.pkl")

    if st.button(
        "Predict Cancellation",
        type="primary",
        key="predict_cancellation_button"
    ):
        # Calculate engineered features
        previous_cancellations_rate = (
            previous_cancellations / previous_orders
            if previous_orders > 0 else 0
        )

        shipping_cost_pct = (
            shipping_cost / order_value * 100
            if order_value > 0 else 0
        )

        # Prepare model input using the exact training columns
        sample_order = pd.DataFrame(
            0,
            index=[0],
            columns=xgb_final.feature_names_in_
        )

        sample_order["quantity"] = quantity
        sample_order["order_total"] = order_value
        sample_order["unit_price"] = order_value / quantity
        sample_order["discount_pct"] = discount_pct
        sample_order["shipping_cost"] = shipping_cost
        sample_order["days_to_deliver"] = days_to_deliver
        sample_order["previous_orders"] = previous_orders
        sample_order["previous_cancellations"] = previous_cancellations
        sample_order["previous_cancellations_rate"] = previous_cancellations_rate
        sample_order["shipping_cost_pct"] = shipping_cost_pct
        sample_order["is_first_order"] = int(customer_type == "New")

        # Set categorical features using training column names
        payment_column = f"payment_method_{payment_method}"
        category_column = f"product_category_{product_category}"

        if payment_column in sample_order.columns:
            sample_order[payment_column] = 1

        if category_column in sample_order.columns:
            sample_order[category_column] = 1

        # Predict cancellation probability
        probability = xgb_final.predict_proba(sample_order)[0, 1]
        probability_pct = probability * 100

        # Assign risk level and recommended action
        if probability_pct < 40:
            risk_level = "Low"
            action = "Process order normally"
        elif probability_pct < 70:
            risk_level = "Medium"
            action = "Send order confirmation reminder"
        else:
            risk_level = "High"
            action = "Contact customer for order confirmation"

        # Display ONLY the five required outputs
        st.divider()
        st.subheader("Prediction Results")

        order_iD ="ORD-"+ uuid.uuid4().hex[:6].upper()

        st.write("Order ID:", order_iD)
        st.write("Cancellation Probability:", f"{probability_pct:.2f}%")
        st.write("Risk Level:", risk_level)
        st.write("Key Factors:", "Delivery time, customer history, order value, payment method and product category")
        st.write("Recommended Action:", action)

elif page == "Bulk Prediction":
        st.title("Bulk Order Cancellation Prediction")

        st.write(
            "Upload an Excel or CSV file containing multiple orders "
            "to predict cancellation probability and risk level."
        )
        # 1. LOAD TRAINED MODEL
        @st.cache_resource
        def load_model():
            model = joblib.load("xgboost_undersampled_model.pkl")
            feature_columns = joblib.load("xgboost_undersampled_feature_columns.pkl")
            return model, feature_columns
        try:
            model, feature_columns = load_model()
        except Exception as e:
            st.error(f"Unable to load the trained model: {e}")
            st.stop()
        # 2. UPLOAD BULK ORDERS FILE
        uploaded_file = st.file_uploader("Upload Bulk Orders File",type=["csv", "xlsx","xls"],
                                         key="bulk_orders_upload")
        if uploaded_file is not None:
            try:
                # Read uploaded file
                if uploaded_file.name.lower().endswith(".csv"):
                    uploaded_df = pd.read_csv(uploaded_file)
                else:
                    uploaded_df = pd.read_excel(uploaded_file)
                if uploaded_df.empty:
                    st.warning("The uploaded file contains no orders.")
                    st.stop()
                st.subheader("Uploaded Orders")
                st.write(f"**Total Orders:** {len(uploaded_df)}")
                st.dataframe(uploaded_df,use_container_width=True)
                # 3. PREDICT CANCELLATION
                if st.button("Predict Cancellation Risk",key="bulk_predict_button"):
                    # Validate required customer column
                    if "customer_type" not in uploaded_df.columns:
                        st.error(
                                "The uploaded file must contain "
                                "a 'customer_type' column.")
                        st.stop()
                    uploaded_df.columns = (uploaded_df.columns.str.strip().str.lower())
                    base_order_id ="ORD-D16C"
                    order_ids = [f"{base_order_id}{i:02}"for i in range(1,len(uploaded_df)+1)]
                    order_ids = pd.Series(order_ids)
                    # Prepare features
                    X = uploaded_df.copy()
                    # Encode categorical columns
                    X = pd.get_dummies(X)
                    # Match training feature names and order
                    X = X.reindex(columns=feature_columns,fill_value=0)
                    # Ensure numeric feature values
                    X = X.apply(pd.to_numeric,errors="coerce")
                    # Stop if missing values remain
                    if X.isnull().any().any():
                        st.error(
                            "The uploaded data contains missing or "
                            "invalid values in model features. "
                            "Apply the same missing-value preprocessing "
                            "used during training before predicting."
                        )
                        st.stop()
                    # Predict probabilities and classes
                    probabilities = model.predict_proba(X)[:, 1]
                    predictions = model.predict(X)
                    importance =model.feature_importances_
                    top_indices = importance.argsort()[-5:][::-1]
                    top_feature = [feature_columns[i] for i in top_indices]
                    key_factors = [",".join(top_feature)for _ in range(len(uploaded_df))]
                    # 4. CLASSIFY CANCELLATION RISK
                    risk_levels = []

                    for probability in probabilities:
                        if probability < 0.30:
                            risk_levels.append("Low")
                        elif probability < 0.70:
                            risk_levels.append("Medium")
                        else:
                            risk_levels.append("High")

                    # 5. CREATE FINAL FOUR-COLUMN DATAFRAME
                    results_df = pd.DataFrame({
                        "Order ID": order_ids,
                        "Cancellation Probability (%)": (probabilities * 100).round(2),
                        "Cancellation Prediction": ["Cancelled" if prediction == 1
                            else "Not Cancelled"for prediction in predictions],
                        "Risk Level": risk_levels,
                        "Key Factors" : key_factors,
                        "Recommended_Action" : risk_levels
                    })
                    # Save results in session state
                    st.session_state["bulk_prediction_results"] = (
                        results_df)
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")
        # 6. DISPLAY PREDICTION RESULTS
        if "bulk_prediction_results" in st.session_state:
            results_df = st.session_state["bulk_prediction_results"]
            st.divider()
            st.subheader("Cancellation Prediction Results")
            st.write(f"**Total Predictions:** {len(results_df)}")
            st.dataframe(results_df,use_container_width=True,hide_index=True)