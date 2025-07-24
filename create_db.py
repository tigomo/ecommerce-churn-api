from database import Base, engine
from models import Prediction

print("🔧 Création des tables dans la base de données...")
Base.metadata.create_all(bind=engine)
print("✅ Tables créées avec succès.")
