import streamlit as st
import joblib
import os

# Load the model outside of Streamlit's main execution context
model_path = os.path.join(os.path.dirname(__file__), "Models", "naive_bayes.pkl")
rf_model = joblib.load(model_path)

# Define Streamlit app
def main():
    st.title('Sentiment Analysis')

    # Home page
    if st.sidebar.button('Home'):
        st.write('Welcome to Sentiment Analysis!')
        st.write('Please enter your review in the box below to predict sentiment.')

    # Prediction page
    if st.sidebar.button('Predict Sentiment'):
        st.subheader('Predict Sentiment')
        review = st.text_area('Enter your review here:')
        if st.button('Predict'):
            prediction = rf_model.predict([review])[0]
            sentiment = 'Positive' if prediction == 1 else 'Negative'
            st.write('Review:', review)
            st.write('Sentiment:', sentiment)

# Run the app
if __name__ == '__main__':
    main()
