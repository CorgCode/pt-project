#!/usr/bin/env bash
# Configure as the forced command on an environment-specific deployment key.
set -euo pipefail
allowed=${1:?set the environment in the authorized_keys command}
case "$allowed" in staging|production) ;; *) exit 2 ;; esac
read -r command environment revision extra <<< "${SSH_ORIGINAL_COMMAND:-}"
[[ "$command" == pt-release && "$environment" == "$allowed" && -z "$extra" ]]
state="$HOME/.local/share/pt-$environment"
if [[ "$revision" == rollback ]]; then
    revision=$(cat "$state/previous-image")
fi
[[ "$revision" =~ ^ghcr\.io/corgcode/pt-project/api@sha256:[a-f0-9]{64}$ ]]
set -a
source "$HOME/.config/pt-network/$environment.env"
set +a
exec "$HOME/.local/share/pt-network/deploy/release.sh" "$environment" "$revision"
