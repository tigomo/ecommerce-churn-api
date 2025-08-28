# database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# URL de connexion à PostgreSQL (Render) à remplacer par la tienne
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Mau2passe@@ecommerce.cl2gcwskyx77.eu-north-1.rds.amazonaws.com:5432/ecommerce_db")

# Création du moteur SQLAlchemy
engine = create_engine(DATABASE_URL)

# Création de la session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base pour les modèles ORM
Base = declarative_base()
