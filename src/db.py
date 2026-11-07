"""Database helpers: connections, running SQL files and quick queries."""
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import pandas as pd
import psycopg

from config import PG, SQL_DIR


@contextmanager
def connect(autocommit: bool = False):
    conn = psycopg.connect(**PG, autocommit=autocommit)
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


def render_sql(name: str | Path, params: dict | None = None) -> str:
    """Read a .sql file from sql/ and replace {{KEY}} placeholders with SQL string literals,
    e.g. {"MODEL_VERSION": "seg-v1.0"} -> 'seg-v1.0'. Cast in SQL where another type is needed."""
    path = Path(name) if Path(name).is_absolute() else SQL_DIR / name
    sql = path.read_text(encoding="utf-8")
    for key, value in (params or {}).items():
        sql = sql.replace("{{" + key + "}}", "'" + str(value).replace("'", "''") + "'")
    return sql


def run_sql_file(name: str | Path, conn=None, params: dict | None = None) -> None:
    """Execute a whole .sql file (multiple statements) from sql/ in one transaction; see render_sql for params."""
    sql = render_sql(name, params)
    if conn is not None:
        conn.execute(sql)
        return
    with connect() as c:
        c.execute(sql)


def query_df(sql: str, params=None) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    # Postgres numeric (e.g. avg()) arrives as Decimal objects; analysis code expects floats
    for col in df.columns[df.dtypes == object]:
        first = df[col].dropna()
        if len(first) and isinstance(first.iloc[0], Decimal):
            df[col] = df[col].astype(float)
    return df


def scalar(sql: str, params=None):
    with connect() as conn:
        return conn.execute(sql, params).fetchone()[0]
