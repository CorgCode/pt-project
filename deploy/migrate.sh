#!/bin/sh
set -eu
environment=${1:?usage: migrate.sh environment image}
image=${2:?immutable API image required}
case "$environment" in staging|production) ;; *) exit 2 ;; esac
prefix="pt-$environment"
podman run --rm --network "systemd-$prefix-database" \
    --cap-drop=all --security-opt=no-new-privileges \
    --secret "$prefix-migration-url,target=migration-url,uid=10001,gid=10001,mode=0400" \
    --env PT_DATABASE_URL_FILE=/run/secrets/migration-url "$image" \
    alembic -c deploy/alembic.ini upgrade head
podman exec "$prefix-postgres" psql -X -v ON_ERROR_STOP=1 -U postgres -d pt <<'SQL'
GRANT SELECT, INSERT, UPDATE, DELETE ON users, sessions, workspaces, jobs TO pt_app;
SQL
