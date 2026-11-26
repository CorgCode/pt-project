#!/bin/sh
set -eu
# Pass paths only. psql reads the generated hexadecimal passwords inside the session.
psql -X -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
\set owner_password `cat /run/secrets/pg-owner`
\set app_password `cat /run/secrets/pg-app`
CREATE ROLE pt_owner LOGIN PASSWORD :'owner_password';
CREATE ROLE pt_app LOGIN PASSWORD :'app_password';
REVOKE ALL ON DATABASE pt FROM PUBLIC;
GRANT CONNECT ON DATABASE pt TO pt_owner, pt_app;
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE, CREATE ON SCHEMA public TO pt_owner;
GRANT USAGE ON SCHEMA public TO pt_app;
SQL
