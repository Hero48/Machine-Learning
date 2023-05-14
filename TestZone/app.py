import streamlit as st
from database import *
import pandas as pd
import os


st.title('Trading Journal')

st.subheader('Record Trade')
st.image('./images/spykvng.jpg', caption='Forex Image', width=200)


col1, col2 = st.columns(2)

with col1:
    forex_pair = st.selectbox('Forex Pair', ['USDJPY', 'EUROUSD', 'GBPUSD'])
    risk_reward = st.text_input('Risk : Reward')
    session = st.selectbox('Session', ['London', 'Asian', 'NewYork'])
    killzone = st.selectbox('Killzone', ['London', 'NewYork'])

with col2:
    lot_size = st.text_input('Lot Size')
    buy_sell = st.selectbox('Buy/Sell', ['Buy', 'Sell'])
    trade_date = st.date_input('Date')
    confirmation = st.multiselect('Confirmation', ['LQS', 'MSS', 'Displacement', 'FVG'])

   
comments = st.text_area('Comments')
screenshot = st.file_uploader('Screenshot', type=['jpg', 'png', 'jpeg'])



if st.button('Save Trade'):
    image_name = screenshot.name
    with open(os.path.join("./images", image_name), "wb") as f:
        f.write(screenshot.getbuffer())
    # Create a trade object and populate its attributes with the user inputs
    trade = Trade(forex_pair=forex_pair,
                  risk_reward=risk_reward,
                  session=session,
                  killzone=killzone,
                  lot_size=float(lot_size),
                  buy_sell=buy_sell,
                  trade_date=trade_date,
                  confirmation=', '.join(confirmation),
                  comments=comments, 
                  screenshot=image_name)

    # Add the trade object to the database
    with Session(engine) as session:
        session.add(trade)
        session.commit()
    
    st.success('Trade saved')
session = Session(bind=engine)
trades = session.query(Trade).all()

if st.button('SHOW TRADES'):
    data = [[trade.forex_pair, trade.buy_sell, trade.session, trade.lot_size, trade.risk_reward, trade.confirmation, trade.killzone, trade.comments] for trade in trades]
    st.table(data)

    

