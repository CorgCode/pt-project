# Team Git Workflow

## Roles

- @physcorgi — project owner and final code owner.
- @DrMogger — repository administrator.
- Other team members — write access.

## Branches

The `main` branch is the stable integration branch.

Do not work directly in `main`.

Create **one branch per Issue**.

### Branch naming

Use this format:

```text
<github-login>/<type>-<issue-number>-<short-task>
```

Allowed task types:

- `research` — research / literature review;
- `data` — dataset exploration and preprocessing;
- `feature` — implementation of functionality;
- `fix` — bug fixes;
- `docs` — documentation;
- `infra` — repository, CI, Docker, tooling.

Examples:

```text
physcorgi/research-2-host-embedding
DrMogger/infra-7-repository-setup
Tikhon2783/research-4-graph-embeddings
genshpaaack123-byte/data-5-lanl-eda
ElliOrNora/research-6-structural-roles
```

Do not create permanent personal branches such as `daniil`, `dasha`, `dev/dima`, etc.
A branch should exist only for one task and should be deleted after its PR is merged.

Legacy branches created before this convention may keep their names until the current work is merged. All new branches must follow this convention.

## Pull Requests

All changes to `main` should go through a Pull Request.

Before merging:

1. The branch is up to date with `main`.
2. The result matches the linked Issue.
3. The work has been checked locally where applicable.
4. At least one other team member reviews the PR.
5. Review conversations are resolved.
6. The PR contains `Closes #<issue-number>` when it completes an Issue.

Prefer **Squash merge** for task branches.

After merge, delete the task branch.

## Safety

Never commit:

- passwords;
- API tokens;
- private keys;
- VPN configs;
- `.env` files;
- confidential datasets.

Use environment variables and GitHub Secrets for credentials.
