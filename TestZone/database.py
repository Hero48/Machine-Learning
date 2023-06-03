
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.orm import declarative_base, Session
from datetime import datetime

# Define the database schema using SQLAlchemy's declarative_base() function
Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    user = Column(String)
    forex_pair = Column(String(50))
    risk_reward = Column(String(10))
    time = Column(DateTime)
    session = Column(String(50))
    risk_reward = Column(Float)
    buy_sell = Column(String(10))
    trade_date = Column(DateTime)
    confirmation = Column(Text)
    comments = Column(Text)
    screenshot = Column(String(30))
    balance = Column(Float)
    status = Column(String(10))

class Balance(Base):
    __tablename__ = 'balance'

    id = Column(Integer, primary_key=True)
    user = Column(String)
    balance = Column(Float, default=10)

# Create a database engine
engine = create_engine('sqlite:///trades.db')

# # Create the trades table in the database
# Base.metadata.create_all(engine)


