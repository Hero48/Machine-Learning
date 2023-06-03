from database import *
import streamlit as st
import os



st.title('Trading Journal')

st.subheader('Trade Records')

session = Session(bind=engine)
username = st.text_input('', placeholder='ID', key=1)
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
   
   