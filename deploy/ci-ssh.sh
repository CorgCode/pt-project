#!/usr/bin/env bash
set -euo pipefail
environment=${1:?}
image=${2:?}
inference=${3:-}
case "$environment" in staging|production) ;; *) exit 2 ;; esac
[[ "$GITHUB_ACTOR" =~ ^[A-Za-z0-9-]+$ ]]
[[ "$PT_SMOKE_ORIGIN" =~ ^https://[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)+$ ]]
[[ "$PT_SSH_TARGET" =~ ^[a-z_][a-z0-9_-]*@[a-zA-Z0-9.-]+$ ]]
[[ "$image" == rollback || "$image" =~ ^ghcr\.io/[a-z0-9_./-]+@sha256:[a-f0-9]{64}$ ]]
if [[ "$image" == rollback ]]; then
    [[ -z "$inference" ]]
else
    [[ "$inference" =~ ^ghcr\.io/corgcode/pt-project/inference@sha256:[a-f0-9]{64}$ ]]
fi
umask 077
credentials=$(mktemp -d)
trap 'rm -r -- "$credentials"' EXIT
printf '%s\n' "$PT_SSH_KEY" > "$credentials/key"
printf '%s\n' "$PT_SSH_KNOWN_HOSTS" > "$credentials/known_hosts"
printf '%s\n' "${PT_REGISTRY_TOKEN:?}" | ssh -T -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes -o IdentitiesOnly=yes \
    -o UserKnownHostsFile="$credentials/known_hosts" -i "$credentials/key" \
    "$PT_SSH_TARGET" "pt-release $environment $GITHUB_ACTOR $image $inference"
# Check that the HTTPS vhost serves this architecture after the loopback readiness gate.
for path in / /model/ /demo/ /api/ready; do
    curl --fail --silent --show-error --max-time 10 --proto '=https'         --output /dev/null "$PT_SMOKE_ORIGIN$path"
done
