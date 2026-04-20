from typing import Literal, Union
import os
import socket
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.exc import ArgumentError
from snowflake.sqlalchemy import URL
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

SnowflakeRoles = Literal['caboodle_owner', 'ent_forecast_owner']


class SnowflakeDB:
    """Snowflake DB

    Creates a sql alchemy engine and provides a simple query-to-df select

    :param fast_execute: bool as to whether to add a `fast_executemany` engine listner to enable fast `INSERTS`
    :param echo: bool as to whether to print the sql query after execution for debugging (default: False, no echo)
    """

    USER = os.getenv('CABOODLE_SNOW_USER')
    ACCOUNT = os.getenv('CABOODLE_SNOW_ACCOUNT')
    WAREHOUSE: str = 'AA_M_WH'
    KEY_PWD = os.getenv('SNOWFLAKE_KP_AUTH')
    KEY_PATH = os.getenv('SNOWFLAKE_KP_PATH')

    KEY_PRV: bytes = None

    def __init__(self,
                 fast_execute: bool = False,
                 role: SnowflakeRoles = 'caboodle_owner',
                 **kwargs) -> None:

        self.role = role.upper()
        self.echo = kwargs.get('echo', False)

        self.db_engine: sa.engine.Engine = None
        self.proxy: CaboodleProxy = CaboodleProxy()

        self.create_engine_instance()

        if fast_execute:
            self.add_fast_execute()

    def create_engine_instance(self) -> None:

        # get the private key as bytes
        private_key = self.get_key()

        # create a snowflake engine as schema agnostic
        connection_url = URL(
            account=self.ACCOUNT,
            user=self.USER,
            warehouse=self.WAREHOUSE,
            role=self.role,
        )

        # get proxy details and define settings
        proxy_str = CaboodleProxy().get_proxy(include_env=False, return_str=True)

        if proxy_str:
            proxy_host, proxy_port = proxy_str.replace('http://', '').split(':')
            proxy_settings = {
                "proxy_host": proxy_host,
                "proxy_port": proxy_port,
            }
        else:
            proxy_settings = {}

        # creat connection settings
        connect_args = {
            "private_key": private_key,
            "client_session_keep_alive": True,
            "login_timeout": 30,
            "network_timeout": 60,
            **proxy_settings,  # merge proxy into connect_args
        }

        self.db_engine: sa.engine.Engine = sa.create_engine(
            url=connection_url,
            connect_args=connect_args,
            pool_pre_ping=True,
            echo=self.echo,
        )

    @classmethod
    def get_key(cls) -> bytes:
        if cls.KEY_PRV:
            return cls.KEY_PRV

        # get the file and read the contents
        with open(cls.KEY_PATH, mode='rb') as key_file:
            key_contents = key_file.read()

        # decode the file with key auth
        private_key = serialization.load_pem_private_key(
            key_contents,
            password=cls.KEY_PWD.encode() if cls.KEY_PWD else None,
            backend=default_backend()
        )

        # encode to der and serialize o bytes
        private_key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        return private_key_bytes

    def add_fast_execute(self, set_active: bool = True):
        """Add Fast Execute

        Adds a 'listner' to SqlAlchemy's `execute` function on the `engine` to enable `fast_executemany`

        :param set_active: bool as to whether to activate the fast execute, default is True
        """

        # pylint: disable=R0913 # too many arguments, not changeable
        # pylint: disable=W0613 # unused arguments, not changeable
        @sa.event.listens_for(self.db_engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
            if executemany:
                cursor.fast_executemany = set_active

    def str_to_sqltext(self, sql_str: str) -> TextClause:
        """Str to Sql (Alchemy) Text

        Converts a str to a Sql Alchemy `text` obj

        :param sql_str: the str to be converted        
        """

        sql_text = sa.text(sql_str)

        return sql_text

    async def select(self, sql: str, params: Union[dict, None] = None) -> pd.DataFrame:
        """(SQL) SELECT (Async)

        A sql SELECT query that returns a Pandas df

        :param sql: a sql str query with `:` placeholders
        :param params: an optional dict of parameters to add to the sql, default `None`
        """

        if params is None:
            params = {}

        # initiate engine session with auto-comment
        with self.db_engine.begin() as conn:
            sa_query = sa.text(sql)

            # check if value is list, if so, then expand as bindparam
            for k, v in params.items():
                if isinstance(v, list):
                    sa_query = sa_query.bindparams(sa.bindparam(k, expanding=True))

            # use pandas to collect result to df
            df = pd.read_sql(sa_query, conn, params=params)

        return df

    def sync_select(self, sql: str, params: Union[dict, None] = None, **kwargs) -> pd.DataFrame:
        """(SQL) SELECT (Sync)

        A sql SELECT query that returns a Pandas df

        :param sql: a sql str query with `:` placeholders
        :param params: an optional dict of parameters to add to the sql, default `None`
        """

        if params is None:
            params = {}

        # initiate engine session with auto-comment
        with self.db_engine.begin() as conn:
            sa_query = sa.text(sql)

            # check if value is list, if so, then expand as bindparam
            for k, v in params.items():
                if isinstance(v, list):
                    try:
                        sa_query = sa_query.bindparams(sa.bindparam(k, expanding=True))
                    except ArgumentError:
                        pass

            # use pandas to collect result to df
            df = pd.read_sql(sa_query, conn, params=params, **kwargs)

        return df

    def truncate_table(self, table_name: str, database: Literal['ENT_FORECAST_PRD', 'EDW_CBD_DEV'], schema: Literal['STAGING'], con: sa.Connection = None) -> bool:
        """Truncate Table

        Truncates a table based on name and schema, allows for existing connection

        :param table_name: the name of the table to truncate
        :param schema: the name of the schema
        :param con: if specified an existing connection is used, default `None` starts a new connection
        """

        try:
            # define prohibited query terms
            prohibited = [' ', ';', 'select', 'drop', 'null', 'with', 'exec', 'execute', 'create', 'truncate', 'replace']

            # check table name for prohibited terms
            if any(x in database.lower() for x in prohibited):
                raise ValueError('Invalid warehouse name specified')

            # check schema for prohibited terms
            if any(x in schema.lower() for x in prohibited):
                raise ValueError('Invalid schema specified')

            # check table name for prohibited terms
            if any(x in table_name.lower() for x in prohibited):
                raise ValueError('Invalid table name specified')

            # create the sql
            _trunc_sql = f'TRUNCATE TABLE {database}.{schema}.{table_name}'

            # create the query
            _query = self.str_to_sqltext(sql_str=_trunc_sql)

            # if a connection is provided use it, otherwise start a new
            if con:
                con.execute(_query)
            else:
                with self.db_engine.begin() as conn:
                    conn.execute(_query)

            return True

        except Exception:
            return False


class CaboodleDB:
    """Caboodle DB

    Creates a sql alchemy engine and provides a simple query-to-df select

    :param production: bool as to whether to use production or staging server (default: True, production)
    :param win_auth: bool as to whether to use windows active login as user or caboodle_app (default: True, win user)
    :param fast_execute: bool as to whether to add a `fast_executemany` engine listner to enable fast `INSERTS`
    :param echo: bool as to whether to print the sql query after execution for debugging (default: False, no echo)
    """
    # sql account db settings
    PRD_SERVER = 'caboodle-01a'
    STG_SERVER = 'entstgpycb01.nonprod-evt.aws'
    DATABASE = 'evtcaboodle'

    # sql connection string settings
    TRUSTED_DB_CONNECTION = 'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};trusted_connection=yes'
    APP_DB_CONNECTION = 'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={user};PWD={password}'

    # sql user settings
    CABOODLE_USER = os.getenv('CABOODLE_SQL_USER')
    CABOODLE_KEY = os.getenv('CABOODLE_SQL_KEY')

    def __init__(self,
                 production: bool = True,
                 win_auth: bool = True,
                 fast_execute: bool = False,
                 **kwargs) -> None:

        self.production = production
        self.win_auth = win_auth
        self.echo = kwargs.get('echo', False)

        self.db_engine: sa.engine.Engine = None

        self.create_engine_instance()

        if fast_execute:
            self.add_fast_execute()

    def create_engine_instance(self) -> None:
        """Create Engine Instance

        Creates a SQL db engine        
        """
        # if production then return prod server path otherwise return staging server path
        if self.production:
            server = self.PRD_SERVER
        else:
            server = self.STG_SERVER

        # if win_user then return trusted connection otherwise use caboodle app user with key
        if self.win_auth:
            _connection_str = self.TRUSTED_DB_CONNECTION.format(
                server=server,
                database='evtcaboodle',
            )
        else:
            _connection_str = self.APP_DB_CONNECTION.format(
                server=server,
                database='evtcaboodle',
                user=self.CABOODLE_USER,
                password=self.CABOODLE_KEY
            )

        # convert string to url and add connector
        connection_url = sa.engine.URL.create("mssql+pyodbc", query={"odbc_connect": _connection_str})

        # establish engine
        self.db_engine: sa.engine.Engine = sa.create_engine(url=connection_url, echo=self.echo)

    def add_fast_execute(self, set_active: bool = True):
        """Add Fast Execute

        Adds a 'listner' to SqlAlchemy's `execute` function on the `engine` to enable `fast_executemany`

        :param set_active: bool as to whether to activate the fast execute, default is True
        """

        # pylint: disable=R0913 # too many arguments, not changeable
        # pylint: disable=W0613 # unused arguments, not changeable
        @sa.event.listens_for(self.db_engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
            if executemany:
                cursor.fast_executemany = set_active

    async def select(self, sql: str, params: Union[dict, None] = None) -> pd.DataFrame:
        """(SQL) SELECT (Async)

        A sql SELECT query that returns a Pandas df

        :param sql: a sql str query with `:` placeholders
        :param params: an optional dict of parameters to add to the sql, default `None`
        """

        if params is None:
            params = {}

        # initiate engine session with auto-comment
        with self.db_engine.begin() as conn:
            sa_query = sa.text(sql)

            # check if value is list, if so, then expand as bindparam
            for k, v in params.items():
                if isinstance(v, list):
                    sa_query = sa_query.bindparams(sa.bindparam(k, expanding=True))

            # use pandas to collect result to df
            df = pd.read_sql(sa_query, conn, params=params)

        return df

    def sync_select(self, sql: str, params: Union[dict, None] = None) -> pd.DataFrame:
        """(SQL) SELECT (Sync)

        A sql SELECT query that returns a Pandas df

        :param sql: a sql str query with `:` placeholders
        :param params: an optional dict of parameters to add to the sql, default `None`
        """

        if params is None:
            params = {}

        # initiate engine session with auto-comment
        with self.db_engine.begin() as conn:
            sa_query = sa.text(sql)

            # check if value is list, if so, then expand as bindparam
            for k, v in params.items():
                if isinstance(v, list):
                    sa_query = sa_query.bindparams(sa.bindparam(k, expanding=True))

            # use pandas to collect result to df
            df = pd.read_sql(sa_query, conn, params=params)

        return df

    async def muli_select(self, sql: Union[str, TextClause], params: dict, labels: Union[list, None] = None) -> dict[str: pd.DataFrame]:
        """Multi (SQL) SELECT (Async) 

        A sql SELECT query that includes multiple statments, returns a dict of Pandas df

        :param sql: a sql str query with `:` placeholders
        :param params: a dict of parameters to add to the sql
        :param labels: an optional list of labels to use as dict keys to identify the dfs, if not supplied then an int is used
        """

        # complile the query and params, required for cursor us
        sql_text = self.compile_query(sql, params)

        # get raw connection from engine
        connection = self.db_engine.raw_connection()

        # if no labels specified then create empty list
        if labels is None:
            labels = []

        # set datasets for return
        datasets = {}

        # get each recordset from the query
        try:
            # create a cursor and execute the query
            cursor = connection.cursor()
            cursor.execute(sql_text.text)

            # while next recordsets exists loop
            x = 0
            while 1:
                # if the labels run out then resort to loop id
                if x < len(labels):
                    label = labels[x]
                else:
                    label = x

                # get columns (column_name, type_, ignore_, ignore_, ignore_, null_ok, column_flags)
                column_names = [col[0] for col in cursor.description]
                data = []

                # fetch all records and append to data
                for row in cursor.fetchall():
                    data.append({name: row[i] for i, name in enumerate(column_names)})

                # append dataset as a df
                datasets[label] = pd.DataFrame(data)

                # next loop count
                x += 1

                # if no next dataset then break
                if cursor.nextset() is None:
                    break

                # nextset() doesn't seem to be sufficiant to tell the end.
                if cursor.description is None:
                    break

        finally:
            # end the cursor connection
            connection.close()

        return datasets

    def truncate_table(self, name: str, schema: Literal['stg', 'dbo'], con: sa.Connection = None) -> bool:
        """Truncate Table

        Truncates a table based on name and schema, allows for existing connection

        :param table_name: the name of the table to truncate
        :param schema: the name of the schema
        :param con: if specified an existing connection is used, default `None` starts a new connection
        """

        try:
            # define prohibited query terms
            prohibited = [' ', ';', 'select', 'drop', 'null', 'with', 'exec', 'execute', 'create', 'truncate']

            # check schema for prohibited terms
            if any(x in schema.lower() for x in prohibited):
                raise ValueError('Invalid schema specified')

            # check table name for prohibited terms
            if any(x in name.lower() for x in prohibited):
                raise ValueError('Invalid table name specified')

            # create the sql
            _trunc_sql = f'TRUNCATE TABLE {schema}.{name}'

            # create the query
            _query = self.str_to_sqltext(sql_str=_trunc_sql)

            # if a connection is provided use it, otherwise start a new
            if con:
                con.execute(_query)
            else:
                with self.db_engine.begin() as conn:
                    conn.execute(_query)

            return True

        except Exception:
            return False

    def str_to_sqltext(self, sql_str: str) -> TextClause:
        """Str to Sql (Alchemy) Text

        Converts a str to a Sql Alchemy `text` obj

        :param sql_str: the str to be converted        
        """

        sql_text = sa.text(sql_str)

        return sql_text

    def compile_query(self, sql_str: Union[str, TextClause], params: dict = None) -> TextClause:
        """Compile Query

        Compiles a sql str (with `:` param placeholders) and parameters to a Sql Alchemy `text` obj.
        Use `compile_query(...).text` to return the compiled str only

        :param sql_str: the str with `:` placeholders to be converted
        :param params: a dict of sql parameters to add to the sql str
        """

        if not params:
            params = {}

        if isinstance(sql_str, str):
            sql_str = sa.text(sql_str)

        if isinstance(sql_str, TextClause):
            sql_bind = sql_str.bindparams(**params).compile(compile_kwargs={"literal_binds": True})
            sql_compile = sa.text(sql_bind.string)
        else:
            sql_compile = sa.text('')

        return sql_compile


class CaboodleProxy:
    """Caboodle Proxy

    Sets the environment proxy variables and/or returns a dict of proxy addresses
    """

    PROTOCOLS = ['http', 'https']
    PROXY_URLS = {
        'caboodle-01a': 'http://proxy.transit.evt.aws:3128',
        'ggndsk005': 'http://evtproxy.ahlnet.local:8080',
    }

    def __init__(self) -> None:
        pass

    def get_proxy(self, include_env: bool = False, return_str: bool = False) -> Union[dict, str]:
        """Get Proxy

        Checks the system environment for `CABOODLE_PROXY` and the class with proxies
        mapped by host name and returns a dict of the proxy.

        :param include_env: bool as to whether to set the system environment variables with the proxy, default is `False`
        :param return_str: bool whether to return the proxy path as a str, default `False` is a dict
        """

        proxy_dict = {}

        # get system environment proxy if it exists
        env_proxy = os.environ.get('CABOODLE_PROXY', False)

        # get mapped by host proxy if it exists
        _hostname = socket.gethostname()
        host_proxy = CaboodleProxy.PROXY_URLS.get(_hostname.lower(), False)

        # selec env over host proxy, return nothing if no proxy
        if env_proxy:
            proxy_url = env_proxy
        elif host_proxy:
            proxy_url = host_proxy
        else:
            return '' if return_str else proxy_dict

        # set the proxy for each protocol in upper and lower cases
        for p in CaboodleProxy.PROTOCOLS:
            if include_env:
                _protocol = p + '_proxy'
                os.environ[_protocol.lower()] = proxy_url
                os.environ[_protocol.upper()] = proxy_url

            proxy_dict[p] = proxy_url

        if return_str:
            return proxy_dict.get('http', '')

        return proxy_dict

    def clear_proxy_env(self) -> None:
        """Clear Proxy Environment (Variable)

        Clears the system proxy environment variables if they are set
        """

        # loop the protocols and remove both lower and upper cases
        for p in CaboodleProxy.PROTOCOLS:
            _protocol = p + '_proxy'
            if os.environ.get(_protocol.lower()):
                os.environ.pop(_protocol.lower())

            if os.environ.get(_protocol.upper()):
                os.environ.pop(_protocol.upper())
