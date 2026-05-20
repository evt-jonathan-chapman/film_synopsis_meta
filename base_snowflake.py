"""
base_snowflake.py — minimal Snowflake helper.

Vendored copy of the SnowFlakeBase class previously imported from
/Users/jonathanchapman/Documents/git/evt_back_up/base/base_snowflake.py.
Brought in-repo so callers can `from base_snowflake import SnowFlakeBase`
without injecting absolute paths onto sys.path.

Account / user are still hard-coded against EVT's Snowflake — if this repo
ever needs to run as a different user, switch to env vars.
"""

import pandas as pd
from sqlalchemy import create_engine, text
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization


class SnowFlakeBase:

    def __init__(self, warehouse, schema, database):
        self.engine = None
        self.warehouse = warehouse
        self.database = database
        self.schema = schema

    def create_snowflake_connection(self, private_key_path):
        with open(private_key_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None,
                backend=default_backend(),
            )

        private_key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        self.engine = create_engine(
            "snowflake://jonathan_chapman@evt.com@mm31132.ap-southeast-2",
            connect_args={
                "private_key": private_key_bytes,
                "warehouse": self.warehouse,
                "database": self.database,
                "schema": self.schema,
            },
        )

    def execute(self, sql_query):
        with self.engine.connect() as conn:
            conn.execute(text(sql_query))
            conn.commit()

    def return_query_output(self, sql_query):
        df = pd.read_sql(sql_query, self.engine)
        df.columns = df.columns.str.lower()
        return df
