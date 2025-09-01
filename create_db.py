# create_db.py
from database import engine, Base
from models import Prediction

def create_db():
    print("🚀 Initialisation de la base de données...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables créées avec succès !")

if __name__ == "__main__":
    create_db()
