#!/usr/bin/env bash
# Install or switch a prebuilt API image. Schema revisions must support both releases.
set -euo pipefail
environment=${1:?usage: release.sh staging|production image@sha256:digest}
export PT_API_IMAGE=${2:?immutable API image required}
case "$environment" in staging|production) ;; *) exit 2 ;; esac
cd "$(dirname "$0")/.."
prefix=pt-$environment
umask 077
state="$HOME/.local/share/$prefix"
mkdir -p "$state"
exec 8>"$state/release.lock"
flock -n 8
units="$HOME/.config/containers/systemd"
mkdir -p "$units"
rendered=$(mktemp -d "$state/render.XXXXXX")
trap 'rm -r -- "$rendered"' EXIT
: "${PT_STATIC_ROOT:?public document root symlink required}"
static_parent=$(dirname "$PT_STATIC_ROOT")
digest=${PT_API_IMAGE##*@sha256:}
python3 deploy/quadlet/render.py "$environment" "$rendered"
python3 deploy/provision-secrets.py "$environment"
previous=$(readlink "$PT_STATIC_ROOT" || true)
if [[ -e "$PT_STATIC_ROOT" && ! -L "$PT_STATIC_ROOT" ]]; then
    echo "Static root must be a release symlink" >&2
    exit 1
fi
podman pull "$PT_API_IMAGE"
revision="$static_parent/$prefix-$digest"
if [[ ! -d "$revision" ]]; then
    temporary_assets=$(mktemp -d "$static_parent/$prefix-assets.XXXXXX")
    # The primary container user exports to stdout. Host tar owns the output files.
    podman run --rm --network=none --cap-drop=all "$PT_API_IMAGE" \
        sh -ec 'python -m deploy.export-static /tmp/static && tar -C /tmp/static -cf - .' |
        tar --no-same-owner -xf - -C "$temporary_assets"
    chmod -R a+rX "$temporary_assets"
    mv -T "$temporary_assets" "$revision"
fi
if [[ -f "$units/$prefix-api.container" ]]; then
    cp "$units/$prefix-api.container" "$rendered/previous-api"
fi
rollback() {
    if [[ -f "$rendered/previous-api" ]]; then
        cp "$rendered/previous-api" "$units/$prefix-api.container"
        systemctl --user daemon-reload
        systemctl --user restart "$prefix-api.service"
    else
        systemctl --user stop "$prefix-api.service" || true
    fi
}
install -m 0644 "$rendered"/*.network "$rendered"/*.volume "$rendered/$prefix-postgres.container" "$units"
systemctl --user daemon-reload
systemctl --user start "$prefix-postgres.service"
for attempt in {1..60}; do
    if podman exec "$prefix-postgres" pg_isready -h 127.0.0.1 -U postgres -d pt; then
        break
    fi
    sleep 2
done
podman exec "$prefix-postgres" pg_isready -h 127.0.0.1 -U postgres -d pt
deploy/migrate.sh "$environment" "$PT_API_IMAGE"
if [[ -f "$rendered/$prefix-inference.container" ]]; then
    install -m 0644 "$rendered/$prefix-inference.container" "$units"
    systemctl --user daemon-reload
    systemctl --user start "$prefix-inference.service"
fi
install -m 0644 "$rendered/$prefix-api.container" "$units"
systemctl --user daemon-reload
if ! systemctl --user restart "$prefix-api.service"; then
    rollback
    exit 1
fi
healthy=false
for attempt in {1..30}; do
    if curl -fsS --max-time 2 -H "Host: $PT_HOST" "http://127.0.0.1:$PT_API_PORT/api/ready" >/dev/null; then
        healthy=true
        break
    fi
    sleep 1
done
if [[ "$healthy" != true ]]; then
    rollback
    exit 1
fi
link="$static_parent/.$prefix-link-$$"
if ! ln -s "$revision" "$link" || ! mv -Tf "$link" "$PT_STATIC_ROOT"; then
    rollback
    exit 1
fi
if [[ -f "$state/current-image" ]]; then
    cp "$state/current-image" "$state/previous-image"
fi
printf '%s\n' "$PT_API_IMAGE" > "$state/current-image"
install -m 0644 "$rendered/$prefix.conf" "$state/nginx.conf"
printf 'API ready. Previous assets: %s\n' "$previous"
