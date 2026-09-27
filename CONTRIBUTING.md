# Team Git Workflow

## Roles

- @physcorgi — project owner and final code owner.
- Dima — repository administrator.
- Other team members — write access.

## Branches

The `main` branch is the stable integration branch.

Do not work directly in `main`.

Create a separate branch for each task:

```
<name>/<type>-<short-task>
```

Examples:

```
daniil/feature-data-loader
dasha/research-node2vec
tikhon/feature-clustering
dima/ci
```

## Pull Requests

All changes to `main` should go through a Pull Request.

Before merging:

1. The branch is up to date with `main`.
2. The code runs locally.
3. Tests pass when tests are available.
4. At least one other team member reviews the PR.
5. Review conversations are resolved.

Prefer squash merge for feature branches.

## Safety

Never commit:

- passwords;
- API tokens;
- private keys;
- VPN configs;
- .env files;
- confidential datasets.

Use environment variables and GitHub Secrets for credentials.
