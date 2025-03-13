# db.py
import sqlite3
from contextlib import contextmanager

DATABASE = "food_app.db"

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    try:
        yield conn
    finally:
        conn.close()

def query_db(query, args=(), one=False):
    """Query the database and return results"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(query, args)
        rv = cur.fetchall()
        conn.commit()
        return (dict(rv[0]) if rv else None) if one else [dict(row) for row in rv]

def insert_db(query, args=()):
    """Insert into database and return the last row id"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(query, args)
        conn.commit()
        return cur.lastrowid