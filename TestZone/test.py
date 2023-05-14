import requests
import streamlit as st




from forex_python.converter import CurrencyRates

c = CurrencyRates()



st.title('Test OF API')
if st.button('Show data'):
    

    rate = c.get_rates('USD', 'EUR')
    text = 'Latest exchange rate: 1 USD = {:.2f} EUR'.format(rate)
    st.write(text)

