from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

if __name__ == '__main__':
    with app.app_context():
        try:
            db.session.execute(text('SELECT 1'))
            print("✅ MySQL Connected Successfully!")
        except Exception as e:
            print(f"❌ Connection Failed: {e}")