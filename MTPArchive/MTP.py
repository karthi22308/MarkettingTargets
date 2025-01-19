import streamlit as st
import numpy as np
import pandas as pd
from openai import AzureOpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import io

# Load trained model and data
df = pd.read_csv("bank_customers_updated.csv")
features = ["Age", "Income", " TransactionFrequency", " AverageTransactionValue", "YearsWithBank", "MostTransactionArea", "CreditCard"]

# Preprocess data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Age", "Income", " TransactionFrequency", " AverageTransactionValue", "YearsWithBank"]),
        ("cat", OneHotEncoder(), ["MostTransactionArea"]),
    ]
)
X_preprocessed = preprocessor.fit_transform(df)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_preprocessed)

def generate_response(input):
    client = AzureOpenAI(api_version='2024-06-01', azure_endpoint='https://hexavarsity-secureapi.azurewebsites.net/api/azureai', api_key='4ceeaa9071277c5b')
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role': 'user', 'content': input}],
        temperature=0.7,
        max_tokens=2560,
        top_p=0.6,
        frequency_penalty=0.7
    )
    return res.choices[0].message.content

# Streamlit UI
st.title("Bank Marketing Service Predictor")

# Mode Selection
mode = st.radio("Select Mode", ("Form Mode", "Export Mode"))

if mode == "Form Mode":
    with st.form("customer_form"):
        # Inputs
        age = st.slider("Age", 18, 65, 30)
        income = st.number_input("Income (in USD)", min_value=1000, max_value=200000, value=50000)
        trans_freq = st.slider("Transaction Frequency (per month)", 1, 20, 10)
        avg_trans_value = st.number_input("Average Transaction Value (in USD)", min_value=10, max_value=1000, value=200)
        years_with_bank = st.slider("Years with Bank", 1, 30, 5)
        most_trans_area = st.multiselect(
            "Most Transactions Area",
            ["Food", "Entertainment", "Travel", "Shopping", "Utilities"],
            default=["Food"]
        )
        creditcard = st.checkbox('Credit card')

        # Submit button
        submitted = st.form_submit_button("Predict")

        if submitted:
            # Encode the input
            input_data = pd.DataFrame({
                "Age": [age],
                "Income": [income],
                " TransactionFrequency": [trans_freq],
                " AverageTransactionValue": [avg_trans_value],
                "YearsWithBank": [years_with_bank],
                "MostTransactionArea": [most_trans_area[0] if most_trans_area else "Food"],  # Default to "Food" if none selected
                "CreditCard": [1 if creditcard else 0]
            })
            input_preprocessed = preprocessor.transform(input_data)
            cluster = kmeans.predict(input_preprocessed)[0]

            # Display result
            prediction = ' '
            st.write(f"Predicted Cluster: {'Luxury Spending' if cluster == 1 else 'Basic Spending'}")
            st.write("Suggested Services:")
            if cluster == 1:
                if creditcard:
                    st.write("- Premium Travel Packages")
                else:
                    st.write("- Premium Credit Card Offers\n- Exclusive Travel Packages")
                prediction = "Premium Credit Card Offers- Exclusive Travel Packages"
            elif cluster == 0:
                st.write("- Basic Savings Account\n- Budget Management Tools")
                prediction = "Basic Savings Account- Budget Management Tools"

            prompt = (
                "your a banker assistant for wintrust bank, based on the data and a k- clustering ML algorithm result it suggested that we can offer"
                + str(prediction) +
                " to customer suggest some advices to this customer based on his most spending area like if more on travel means suggest something accordingly and give some financial advice.  no salutations only content"
                + str(age) + " income:" + str(income) + " Transaction Frequency:" + str(trans_freq) +
                " Average Transaction Value (in USD):" + str(avg_trans_value) + " years_with_bank :" + str(years_with_bank) +
                " most transaction area:" + str(most_trans_area) + "everything should be within wintrust banks services"
            )
            gen_ai_out = generate_response(prompt)
            st.markdown(gen_ai_out)

elif mode == "Export Mode":
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Read the uploaded Excel file
        input_df = pd.read_excel(uploaded_file)

        # Preprocess and predict
        input_preprocessed = preprocessor.transform(input_df)
        clusters = kmeans.predict(input_preprocessed)

        # Generate AI responses
        ai_responses = []
        for i, row in input_df.iterrows():
            #st.write(row)
            prediction = 'Luxury Spending' if clusters[i] == 1 else 'Basic Spending'
            prompt = (
                "your a banker assistant for wintrust bank, based on the data and a k- clustering ML algorithm result it suggested that we can offer"
                + prediction +
                " to customer suggest some advices to this customer based on his most spending area like if more on travel means suggest something accordingly and give some financial advice.  no salutations only content"
                + str(row['Age']) + " income:" + str(row['Income']) + " Transaction Frequency:" + str(row[' TransactionFrequency']) +
                " Average Transaction Value (in USD):" + str(row[' AverageTransactionValue']) + " years_with_bank :" + str(row['YearsWithBank']) +
                " most transaction area:" + str(row['MostTransactionArea']) + "everything should be within wintrust banks services"
            )
            ai_response = generate_response(prompt)
            ai_responses.append(ai_response)

        # Add AI responses to the dataframe
        input_df['AI_Response'] = ai_responses

        # Create a BytesIO object
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')

        # Write the DataFrame to the Excel file
        input_df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()

        # Reset the buffer position to the beginning
        output.seek(0)

        # Get the processed data
        processed_data = output.getvalue()

        # Save to a file or use as needed
        with open("sample_customers.xlsx", "wb") as f:
            f.write(processed_data)

        print("Sample data with clustering results saved to sample_customers.xlsx")

        st.download_button(
            label="Download Processed Data",
            data=processed_data,
            file_name="processed_customers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
