import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///trade_journal.db')
Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    user = Column(String)
    trade_date = Column(Date)
    symbol = Column(String)
    entry_price = Column(Float)
    exit_price = Column(Float)
    quantity = Column(Integer)

class Balance(Base):
    __tablename__ = 'balance'
    id = Column(Integer, primary_key=True)
    user = Column(String)
    date = Column(Date)
    balance = Column(Float)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


def save_trade(user, trade_date, symbol, entry_price, exit_price, quantity):
    trade = Trade(user=user, trade_date=trade_date, symbol=symbol, entry_price=entry_price,
                  exit_price=exit_price, quantity=quantity)
    session.add(trade)
    session.commit()

user = st.text_input("User")
trade_date = st.date_input("Trade Date")
symbol = st.text_input("Symbol")
entry_price = st.number_input("Entry Price")
exit_price = st.number_input("Exit Price")
quantity = st.number_input("Quantity")

if st.button("Save Trade"):
    save_trade(user, trade_date, symbol, entry_price, exit_price, quantity)
    st.write("Trade saved!")


def display_trade_history(user):
    trades = session.query(Trade).filter_by(user=user).all()
    if trades:
        df = pd.DataFrame([(t.user, t.trade_date, t.symbol, t.entry_price, t.exit_price, t.quantity) for t in trades],
                          columns=['User', 'Trade Date', 'Symbol', 'Entry Price', 'Exit Price', 'Quantity'])
        st.subheader("Trade History")
        st.dataframe(df)
    else:
        st.write("No trades found for the user.")

user = st.text_input("User for Trade History")
if st.button("Display Trade History"):
    display_trade_history(user)
