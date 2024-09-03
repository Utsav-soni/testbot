import streamlit as st

# Function to inject custom CSS


def inject_custom_css():
    st.markdown("""
        <style>
            /* General text color */
            body {
                color: var(--text-color);
            }
            
            /* Header color */
            h1, h2, h3, h4, h5, h6 {
                color: var(--text-color);
            }

            /* Text in markdown */
            .css-1d391kg p {
                color: var(--text-color);
            }
            
            /* Background color of Streamlit widgets */
            .css-1cpxqw2 {
                background-color: var(--background-color) !important;
            }
        </style>
        """, unsafe_allow_html=True)


# Call the function to inject CSS
inject_custom_css()

# Your Streamlit app content
st.title("My Streamlit App")
st.write("This text should be visible in both light and dark modes.")
