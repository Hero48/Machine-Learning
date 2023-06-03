import streamlit as st
from database import *
import pandas as pd
import os


st.title('Trading Journal')

st.subheader('Record Trade')
st.subheader('User')
username = st.text_input('', placeholder='Username')
col1, col2 = st.columns(2)
with col1:
    forex_pair = st.selectbox('Forex Pair', ['USDJPY', 'EUROUSD', 'GBPUSD'])
    risk_reward = st.number_input('Profit Percentage')
    session = st.selectbox('Session', ['London', 'Asian', 'NewYork'])
    status = st.selectbox('Status', ['Won', 'Lost', 'Break Even'])

with col2:
    buy_sell = st.selectbox('Buy/Sell', ['Buy', 'Sell'])
    time = st.time_input('Time')
    trade_date = st.date_input('Date')
    confirmation = st.multiselect('Confirmation', ['LQS', 'MSS', 'Displacement', 'FVG'])


comments = st.text_area('Comments')
screenshot = st.file_uploader('Screenshot', type=['jpg', 'png', 'jpeg'])

if st.button('Save Trade'):
    image_name = screenshot.name
    with open(os.path.join("./images", image_name), "wb") as f:
        f.write(screenshot.getbuffer())
    image_path = os.path.join("./images", image_name)
    st.subheader(image_path)

    # Create a trade object and populate its attributes with the user inputs
    trade = Trade(forex_pair=forex_pair,
                  user=username,
                  risk_reward=risk_reward,
                  time=datetime.now(),
                  session=session,
                  buy_sell=buy_sell,
                  trade_date=trade_date,
                  confirmation=', '.join(confirmation),
                  comments=comments, 
                  screenshot=image_name,
                  balance=0.0,
                  status=status)

    # Add the trade object to the database
    with Session(engine) as session:
        session.add(trade)
        session.commit()

    st.success('Trade saved')

session = Session(bind=engine)
username = st.text_input('', placeholder='Username', key=1)
#query by user
trades = session.query(Trade).filter_by(user=username).all()

def display_trades():
    if trade.status == 'Won':
        st.markdown("""
            <style>
                .trade-card {
                    border: 1px solid #ccc;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    width: 300px;
                    magin: 0 auto;
                    
                }

                .trade-card h3 {
                    margin: 0;
                    font-size: 24px;
                    font-weight: 600;
                    color: #333;
                    text-align: center;
                }

                .trade-card p {
                    margin: 0;
                    font-size: 16px;
                    color: #111;
                }

                .trade-card p:first-child {
                    margin-top: 10px;
                }

                .trade-card p:last-child {
                    margin-bottom: 0;
                }

                .trade-card p:not(:first-child):not(:last-child) {
                    margin-top: 10px;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="trade-card" style="background-color: #c1e6c2 ;">
                <h3>Trade ID: {trade.id}</h3>
                <p>Forex Pair: {trade.forex_pair}</p>
                <p>Buy/Sell: {trade.buy_sell}</p>
                <p>Session: {trade.session}</p>
                <p>Risk/Reward: {trade.risk_reward}</p>
                <p>Trade Date: {trade.trade_date}</p>
                <p>Confirmation: {trade.confirmation}</p>
                <p>Comments: {trade.comments}</p>
            </div>
            <br>
        """, unsafe_allow_html=True)
        if st.button('View Screenshot', key=f'{trade.id}screenshot'):
                if trade.screenshot:
                    st.image(os.path.join("./images", trade.screenshot))
                else:
                    st.markdown("No screenshot available.")
                if st.button('Close screenshot', key=f'{trade.id}close'):
                    pass
    if trade.status == 'Lost':
        st.markdown("""
            <style>
                .trade-card {
                    border: 1px solid #ccc;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    width: 300px;
                    magin: 0 auto;
                    
                }

                .trade-card h3 {
                    margin: 0;
                    font-size: 24px;
                    font-weight: 600;
                    color: #333;
                    text-align: center;
                }

                .trade-card p {
                    margin: 0;
                    font-size: 16px;
                    color: #111;
                }

                .trade-card p:first-child {
                    margin-top: 10px;
                }

                .trade-card p:last-child {
                    margin-bottom: 0;
                }

                .trade-card p:not(:first-child):not(:last-child) {
                    margin-top: 10px;
                }
                </style>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="trade-card" style="background-color: #ffc0c0 ;">
                <h3>Trade ID: {trade.id}</h3>
                <p>Forex Pair: {trade.forex_pair}</p>
                <p>Buy/Sell: {trade.buy_sell}</p>
                <p>Session: {trade.session}</p>
                <p>Risk/Reward: {trade.risk_reward}</p>
                <p>Trade Date: {trade.trade_date}</p>
                <p>Confirmation: {trade.confirmation}</p>
                <p>Comments: {trade.comments}</p>
            </div>
            <br>
            """, unsafe_allow_html=True)
        if st.button('View Screenshot', key=f'{trade.id}screenshot'):
            if trade.screenshot:
                st.image(os.path.join("./images", trade.screenshot))
            else:
                st.markdown("No screenshot available.")
            if st.button('Close screenshot', key=f'{trade.id}close'):
                pass
    st.markdown("---")

        
   






right_column, left_column = st.columns(2)





for trade in trades:
    
    if trade.id % 2 == 0:
        with left_column:
            display_trades()
    else:
        with right_column:
            display_trades()
   
   