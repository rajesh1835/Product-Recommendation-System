import os
from dotenv import load_dotenv

load_dotenv()

def get_database_url():
    """Get database URL, handling Railway's mysql:// prefix"""
    url = os.getenv('DATABASE_URL')
    if url:
        # Railway sometimes gives mysql:// but SQLAlchemy needs mysql+pymysql://
        if url.startswith('mysql://'):
            url = url.replace('mysql://', 'mysql+pymysql://', 1)
        return url
    # Fallback: build from individual vars (for local dev without DATABASE_URL)
    user = os.getenv('DB_USER', 'root')
    password = os.getenv('DB_PASSWORD', '')
    host = os.getenv('DB_HOST', 'localhost')
    db_name = os.getenv('DB_NAME', 'recommender_db')
    return f"mysql+pymysql://{user}:{password}@{host}/{db_name}"

class Config:
    SQLALCHEMY_DATABASE_URI = get_database_url()
    SQLALCHEMY_TRACK_MODIFICATIONS = False