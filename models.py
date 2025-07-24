from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.sql import func
from database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, index=True)
    churn_probability = Column(Float)
    confidence_score = Column(String)
    prediction_type = Column(String)  # "single", "batch", "json_file"
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
