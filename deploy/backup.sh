#!/usr/bin/env bash
set -euo pipefail
environment=${1:?usage: backup.sh staging|production backup|verify}
operation=${2:?}
case "$environment" in staging|production) ;; *) exit 2 ;; esac
prefix=pt-$environment
umask 077
state="$HOME/.local/share/$prefix"
scratch=$(mktemp -d "$state/backup.XXXXXX")
trap 'rm -r -- "$scratch"' EXIT
case "$operation" in
    backup)
        podman exec "$prefix-postgres" pg_dump -U postgres -d pt -Fc --no-owner --no-privileges > "$scratch/pt.dump"
        (
            cd "$scratch"
            restic backup --host "$prefix" --tag postgres pt.dump
        )
        if [[ -n "${PT_BACKUP_ARTIFACTS:-}" ]]; then
            restic backup --host "$prefix" --tag artifacts "$state/datasets" "$state/models" "$state/results"
        fi
        date -u +%FT%TZ > "$state/last-backup"
        ;;
    verify)
        restic dump --host "$prefix" --tag postgres latest /pt.dump > "$scratch/pt.dump"
        # Restore into a fresh, disposable DB with no published port.
        image=${PT_POSTGRES_IMAGE:?immutable PostgreSQL image required}
        container=$(podman run -d --network none --env POSTGRES_HOST_AUTH_METHOD=trust "$image")
        trap 'podman rm -f "$container" >/dev/null; rm -r -- "$scratch"' EXIT
        for attempt in {1..60}; do
            if podman exec "$container" pg_isready -h 127.0.0.1 -U postgres; then break; fi
            sleep 1
        done
        podman exec "$container" pg_isready -h 127.0.0.1 -U postgres
        podman exec -i "$container" pg_restore --exit-on-error --no-owner --no-privileges -U postgres -d postgres < "$scratch/pt.dump"
        podman exec "$container" psql -X -v ON_ERROR_STOP=1 -U postgres -d postgres \
            -c 'SELECT version_num FROM alembic_version' \
            -c 'SELECT count(*) FROM users' -c 'SELECT count(*) FROM workspaces'
        date -u +%FT%TZ > "$state/last-restore-test"
        ;;
    *) exit 2 ;;
esac
