#!/usr/bin/env bash
# Configure as the forced command on an environment-specific deployment key.
set -euo pipefail
allowed=${1:?set the environment in the authorized_keys command}
case "$allowed" in staging|production) ;; *) exit 2 ;; esac
read -r command environment revision inference extra <<< "${SSH_ORIGINAL_COMMAND:-}"
[[ "$command" == pt-release && "$environment" == "$allowed" && -z "$extra" ]]
state="$HOME/.local/share/pt-$environment"
if [[ "$revision" == rollback ]]; then
    [[ -z "$inference" ]]
    revision=$(cat "$state/previous-image")
    inference=$(cat "$state/previous-inference-image")
fi
[[ "$revision" =~ ^ghcr\.io/corgcode/pt-project/api@sha256:[a-f0-9]{64}$ ]]
[[ "$inference" =~ ^ghcr\.io/corgcode/pt-project/inference@sha256:[a-f0-9]{64}$ ]]
set -a
source "$HOME/.config/pt-network/$environment.env"
set +a
# Deployment environments supply a short-lived, package-read-only registry token.
umask 077
auth_dir=$(mktemp -d)
trap 'unlink "$auth_dir/auth.json" 2>/dev/null || true; rmdir "$auth_dir"' EXIT
export REGISTRY_AUTH_FILE="$auth_dir/auth.json"
podman login ghcr.io -u github-actions --password-stdin >/dev/null
export PT_CI_INFERENCE_IMAGE="$inference"
export PT_INFERENCE_IMAGE="${PT_INFERENCE_IMAGE:-$inference}"
"$HOME/.local/share/pt-network/deploy/release.sh" "$environment" "$revision"
