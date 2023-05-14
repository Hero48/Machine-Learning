
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.orm import declarative_base, Session
from datetime import datetime

# Define the database schema using SQLAlchemy's declarative_base() function
Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    forex_pair = Column(String(50))
    risk_reward = Column(String(10))
    session = Column(String(50))
    killzone = Column(String(50))
    lot_size = Column(Float)
    buy_sell = Column(String(10))
    trade_date = Column(DateTime)
    confirmation = Column(Text)
    comments = Column(Text)
    screenshot = Column(String(30))

# Create a database engine
engine = create_engine('sqlite:///trades.db')

# Create the trades table in the database
Base.metadata.create_all(engine)


