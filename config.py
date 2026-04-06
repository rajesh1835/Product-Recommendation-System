import os
from dotenv import load_dotenv

# Ensure .env is loaded from the project root
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'), override=True)

def get_database_url():
    """Get database URL from Railway environment variables.

    Resolution order:
    1. MYSQL_URL  — the full connection URL provided by Railway.
    2. Individual variables: MYSQLHOST, MYSQLPORT, MYSQLUSER,
       MYSQLPASSWORD, MYSQLDATABASE — also provided by Railway.

    Raises EnvironmentError if neither source is available, so the app
    fails fast with a clear message instead of silently trying localhost.
    """
    # 1. Prefer the full URL when Railway provides it.
    url = os.getenv('MYSQL_URL')
    if url:
        # SQLAlchemy requires the mysql+pymysql:// dialect prefix.
        if url.startswith('mysql://'):
            url = url.replace('mysql://', 'mysql+pymysql://', 1)
        return url

    # 2. Build the URL from individual Railway MySQL variables.
    host = os.getenv('MYSQLHOST')
    if not host:
        raise EnvironmentError(
            "Database configuration is missing. "
            "Set MYSQL_URL, or set MYSQLHOST (along with MYSQLPORT, "
            "MYSQLUSER, MYSQLPASSWORD, and MYSQLDATABASE) in the "
            "Railway service environment variables."
        )

    port = os.getenv('MYSQLPORT', '3306')
    user = os.getenv('MYSQLUSER', 'root')
    password = os.getenv('MYSQLPASSWORD', '')
    db_name = os.getenv('MYSQLDATABASE', 'railway')

    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"

class Config:
    SQLALCHEMY_DATABASE_URI = get_database_url()
    SQLALCHEMY_TRACK_MODIFICATIONS = False