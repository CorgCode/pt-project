# PT-05 — Host Behaviour Representation from Network Traffic

Team project for **Positive Technologies / MIPT**.

## Goal

Build a system that learns representations of hosts in a corporate network from network traffic and uses them to:

- find hosts with similar behaviour;
- discover groups / structural roles without manual labels;
- detect suspicious behavioural changes and anomalies;
- explain why hosts are considered similar.

The current project stage is **research**. Each team member performs research in a separate task branch and submits the result through a Pull Request.

## Team

- **Daniil** — @physcorgi
- **Dima** — @DrMogger
- **Tikhon** — @Tikhon2783
- **Danya** — @genshpaaack123-byte
- **Dasha** — @ElliOrNora

## Current stage — Research

The team is currently studying:

- host embeddings and representation learning;
- node2vec / DeepWalk / GraphSAGE;
- structural-role methods such as struc2vec and RolX;
- clustering and dimensionality reduction;
- anomaly and lateral-movement detection;
- the LANL Unified Host and Network Dataset;
- evaluation of unsupervised embeddings and clusters.

Research tasks are tracked in **GitHub Issues** and grouped by milestone.

## Workflow

```text
Issue
  ↓
task branch
  ↓
commits
  ↓
Pull Request
  ↓
review
  ↓
main
```

### Branches

Do not work directly in `main`.

Create one branch for each Issue using:

```text
<github-login>/<type>-<issue-number>-<short-task>
```

Examples:

```text
physcorgi/research-2-host-embedding
DrMogger/infra-7-repository-setup
Tikhon2783/research-4-graph-embeddings
genshpaaack123-byte/data-5-lanl-eda
ElliOrNora/research-6-structural-roles
```

Task branches are temporary and should be deleted after merge.

### Pull Requests

Every change to `main` should go through a Pull Request.

A PR should:

- link the related Issue using `Closes #<issue-number>`;
- describe what was done;
- be reviewed before merge;
- resolve review conversations;
- avoid secrets or confidential data.

Prefer **Squash merge** for task branches.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full team workflow.

## Repository rules

The `main` branch is protected:

- direct changes should go through Pull Requests;
- at least one approval is required;
- CODEOWNER review is required;
- stale approvals are dismissed after new pushes;
- review conversations must be resolved;
- force pushes and branch deletion are blocked.

## Security

Never commit:

- passwords;
- API tokens;
- private keys;
- `.env` files;
- VPN configuration;
- confidential or restricted datasets.

Use environment variables and GitHub Secrets for credentials.

## Next stage

After the research phase, the team will define the MVP architecture and split implementation into separate Issues for:

- data loading and preprocessing;
- feature extraction;
- embeddings;
- clustering;
- anomaly detection;
- evaluation;
- API / interface;
- tests and reproducibility.
