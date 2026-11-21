# PT Network deployment

## Audit and implementation order

The audited VPS is Ubuntu 24.04 with Nginx, Podman 4.9.3, two CPUs and about
3.8 GiB of RAM. The live host routes `mipt.cloud` to loopback port 8845 and
`staging.mipt.cloud` to 8847. The existing `mipt-site` and `mipt-staging`
rootless service accounts already isolate their containers and data.
The Compose files remain available for local development.

1. Use the existing operator FastAPI entry point and SQLite API tests as the
   compatibility baseline. Install PostgreSQL alongside SQLite, rehearse
   Alembic and migration on a disposable database, and verify ownership and
   password compatibility.
2. Export the existing static frontend from the same artifact as the backend.
   Point a separate Nginx test listener at it and validate the API over a
   spare loopback port. Leave the current vhosts and services active.
3. Promote the same scanned image digest to staging, exercise a backed-up
   database, verify offsite restore and alert delivery, then use the protected
   production GitHub environment for approval. The first public switch uses
   an explicit Nginx reload after readiness and recovery checks.
4. Enable real inference only after you have a complete trusted package and
   measured cold-start and peak memory on representative inputs.

## Service boundaries

| Service | Network and state |
| --- | --- |
| Host Nginx | Public HTTPS, static document root, request limits, structured logs and proxy to `127.0.0.1:8945` or `:8947` |
| Rootless FastAPI | One Uvicorn worker, DB-backed auth and sessions, owner-scoped workspaces, dataset metadata and inference job records |
| PostgreSQL 16 | Private Podman network and environment-specific volume. Container starts PostgreSQL as its database user. Separate bootstrap, schema owner and application credentials |
| Python inference | Optional second private network, one worker and one in-flight request. Image-bundled adapter, read-only checkpoint and dataset mounts, writable results mount |
| Prometheus and Grafana | Loopback-only host services. Prometheus reaches the API's bearer-authenticated metrics endpoint directly |
| Restic | Offsite encrypted backups and a scheduled restore into a disposable PostgreSQL container |

`/`, `/model/`, `/product/` and `/demo/` use existing static content.
`/lab/` retains sign-in, save, load and logout. `/demo/` does browser-side
analysis with the existing laboratory assets. The operator keeps the
existing workspace and inference URLs and JSON shapes.
`/api/datasets` lists the saved datasets owned by the session user.
Authenticated inference returns the same response plus `X-PT-Job-ID`, which
can be read at `/api/jobs/{id}` by its owner. Jobs reflect synchronous
inference requests. A worker interruption can leave a running record and
does not retry the request automatically. No broker is needed at this scale.

The browser sends a strict same-origin check for mutating requests. The
session cookie is Secure over HTTPS, HttpOnly and SameSite=Strict. The database
stores only a hash of the session token. The application expects the direct
Nginx proxy and gives public registration an environment switch.
`pt_owner` is used by migrations and imports. `pt_app` receives DML grants
on the application tables and no schema creation permission. Operator roles
allow workspace access, and owner checks are in the queries themselves.

## Bootstrap a separate environment

Use distinct rootless service users and separate config, volumes, registry
credentials, public hostnames and ports. Start with staging. For the VPS in
this audit, reserve 8947 for staging and 8945 for production. Preserve the
existing vhosts until the readiness check and migration are complete.
Administrator actions include enabling lingering, preparing the static
directories (readable by Nginx and writable by the deployment user),
configuring TLS/ACME, and establishing a dedicated SSH principal per
environment. The existing service users have a nologin shell.
Use an administrator-controlled forced-command wrapper that runs as the
corresponding service user, or create dedicated deployment users with a
valid shell and only forced-command keys. Keep the current HTTP-to-HTTPS
and ACME locations.

Install the deployment scripts at `~/.local/share/pt-network` for each
deployment principal. Create `~/.config/pt-network/staging.env` mode 0600
and a separate production file. Example staging values:

```bash
PT_HOST=staging.mipt.cloud
PT_API_PORT=8947
PT_STATIC_ROOT=/var/www/pt-staging/current
PT_CERT=/etc/letsencrypt/live/staging.mipt.cloud/fullchain.pem
PT_CERT_KEY=/etc/letsencrypt/live/staging.mipt.cloud/privkey.pem
PT_POSTGRES_IMAGE=docker.io/library/postgres:16-bookworm@sha256:59ab0bbd4930fcd32dd4389836c0bfbf87431ffa87097c739a71210f7aed58c9
PT_ALLOW_REGISTRATION=0
```

The deploy user needs read access to private GHCR images if the package is
private. Set up a read-only Podman registry login under that user.
Run the first release with a scanned `ghcr.io/corgcode/pt-project/api@sha256:…`
image:

```bash
set -a
. ~/.config/pt-network/staging.env
set +a
~/.local/share/pt-network/deploy/release.sh staging "$IMAGE"
curl --fail -H "Host: $PT_HOST" "http://127.0.0.1:$PT_API_PORT/api/ready"
```

The release script places the candidate vhost at
`~/.local/share/pt-staging/nginx.conf`. Inspect that candidate, install it
in place of the existing TLS vhost only at cutover, run `nginx -t` and
reload Nginx. Keep the old vhost file outside the included directory.
Rollback to a previous application image through the GitHub workflow or
run the release script with the previous digest. Alembic revisions must be
compatible with the previous image through the rollback window. Test CSS,
JS, demo analysis, sign-in, save/load, logout and a database restore on
staging before allowing a production run.

### SQLite cutover

1. Rehearse using a sanitized snapshot. Check the number of accounts and
   workspaces, sign in with an existing account and read a saved workspace.
2. During the production switch, quiesce writes to legacy SQLite before
   taking a consistent copy of the entire volume, including sidecars. Retain
   and encrypt that copy. Start the new database and run the release migration.
3. Mount the copy read-only and run the importer as UID 10001 with a
   readable copy of the snapshot and the schema owner secret.

```bash
podman run --rm --network systemd-pt-production-database \
  --cap-drop=all --security-opt=no-new-privileges \
  --secret pt-production-migration-url,target=migration-url,uid=10001,gid=10001,mode=0400 \
  --env PT_DATABASE_URL_FILE=/run/secrets/migration-url \
  -v "$SNAPSHOT:/legacy:ro" "$IMAGE" python -m app.operator.import_sqlite /legacy
```

Prepare the snapshot permissions inside the rootless user namespace
(`podman unshare`). Legacy IDs, password verifiers and workspace payloads
are carried across in one transaction. Old sessions expire at the switch
and users sign in again. A nonempty destination fails before import.
Compare per-owner counts against the snapshot. Enable writes only after
smoke tests. If writes were accepted on PostgreSQL, preserve and reconcile
them before switching back to SQLite.

For a new environment with registration off, run
`podman exec -it pt-staging-api python -m app.operator.manage_users analyst`.

### Inference and reproducibility

Set `PT_INFERENCE_IMAGE` to the published inference image digest in the
environment. The bundled, checkpoint-free default is `traffic-profile-v1`.
For a small VPS start with `PT_MODEL_MEMORY=512m` and `PT_MODEL_CPUS=1`.
Set `PT_MODEL_VERSION=traffic-volume-v1` to select the smaller profile based
on traffic volume and peer counts. The full profile also uses direction-specific
service frequencies. Both versions reuse `hostquery.store.dataset_from_csv`.
They compute deterministic embeddings from the uploaded window and make no
claims about benchmark scores, future contacts or attack detection.

The signed-in workspace sends flow uploads to
`POST /api/inference/embeddings`; the API calls private `POST /v1/predict`.
Inputs have a 3 MiB CSV limit with 10,000 records and 500 hosts. The model
admits one request at a time. PostgreSQL stores the model output, source
digest and version with the owned job. A saved workspace can reference the job
only if the owner and CSV digest match. Reopening restores the original
embeddings even after switching the deployed model. The raw CSV still supplies
observed contacts and the existing two-window analysis. Precomputed vector CSV
uploads continue through the existing local path.

To integrate a checkpoint-backed model, build a child of the inference image
with a trusted Python module and set `PT_MODEL_ADAPTER` **in that image**.
Expose `load(verified_artifacts, version)`, returning a
`pt-embedding-adapter/1` object with:

- `metadata` containing `method`, `model_version` and `dimensions`
- `preprocess(csv)` producing validated model inputs
- `infer(prepared)` returning `[{id, v, norm}, ...]`
- `validate_output(prepared, items)` enforcing model-specific invariants

The service adds provenance and preserves the `hostquery/1` dataset envelope.
It validates shape, identity and finite nonzero norms before returning, and the
API independently validates the envelope and input digest before persistence.
Put the weights under the read-only model mount with a `manifest.json` such
as `{"version":"v1","files":{"checkpoint.bin":"<sha256>"}}`; set
`PT_MODEL_VERSION` and `PT_MODEL_MANIFEST_SHA256` in the deployment environment.
External adapters cannot start without a verified manifest. Measure the full
Plus package and its RAM before activating it on the VPS.

### Deployment approvals

Create GitHub environments named `staging` and `production`. Restrict
production to `main` and enable required reviewers using GitHub repository
administration. Add `PT_SSH_TARGET`, `PT_SSH_KEY` and pinned
`PT_SSH_KNOWN_HOSTS` to each environment, along with its HTTPS origin as `PT_SMOKE_ORIGIN`.
Use different keys with an
`authorized_keys` entry of the form
`restrict,command="/absolute/path/deploy/ci-entrypoint.sh staging" ...`
or its production equivalent. The interactive administrator key stays
separate. Set `PT_ENABLE_STAGING_DEPLOY=1` and
`PT_ENABLE_PRODUCTION_DEPLOY=1` only after bootstrap.
CI scans the same images it publishes, deploys staging first, and passes
the identical API digest into the approved production job. A manual
rollback uses the same environment restrictions.

### Monitoring and backups

Install Prometheus, Alertmanager and Grafana under dedicated host users.
Bind all three to loopback and use an SSH tunnel for Grafana. Install
`monitoring/prometheus.yml`, `alerts.yml` and the Grafana datasource.
Load per-environment metrics tokens from the relevant Podman secret into
service-readable files under `/etc/prometheus/secrets`. Update the ports
if the environment differs. Configure Alertmanager with your notification
destination and verify receipt before production approval. Use
`up`, `pt_http_requests_total` and `pt_http_request_seconds_bucket` to
inspect availability, errors and latency. Logs are in the user journal and
Nginx access logs. Metrics avoid usernames, request bodies and arbitrary
paths.

Install Restic and initialize a separate encrypted offsite repository for
each environment. Store `RESTIC_REPOSITORY`, `RESTIC_PASSWORD_FILE` and
any storage credential reference in the environment's
`~/.config/pt-network/<environment>-backup.env` with restrictive
permissions. Set `PT_POSTGRES_IMAGE` there for the restore drill. Store
the repository password in a recoverable credential store outside the
VPS. Enable the daily backup and monthly restore timers from
`deploy/quadlet/pt-*.{timer,service}` after installing them in
`~/.config/systemd/user`. The drill restores an encrypted snapshot into
a temporary container and checks schema and row counts. Choose offsite
retention to fit the institution's data policy. Add `PT_BACKUP_ARTIFACTS=1`
when regenerating datasets, checkpoints or results would be impractical.
Keep a runbook copy and image digests offsite as well.

### Current production gate

Require a rehearsal on a sanitized copy of the site's actual SQLite data,
a successful offsite restore and alert delivery, provisioned GitHub
environment restrictions, and a reviewed Nginx change before the first
public cutover. Register the current live deployment as the rollback
target and retain its SQLite snapshot. The bundled traffic profile can be enabled independently of the future Plus package.
