# Deploying Artwork Automation (with PostgreSQL)

The server now stores everything it needs in its own PostgreSQL database: jobs
and their results (so **Previous Runs survive a restart**), users, agent tokens,
and all prompts. Deployment is Docker Compose with two services:

- **`db`** — PostgreSQL 16, data on a named volume `pgdata`, with a healthcheck.
- **`artwork-automation`** — the app (FastAPI + the headed Chromium/noVNC image,
  unchanged). It waits for `db` to be healthy, runs database migrations
  automatically, imports any existing data on first start, then serves on port
  8000 (published to `127.0.0.1:4405`, behind the nginx/authentik vhost on 4400).

> The designer-PC agent is **unchanged**. It still talks only to the server's
> `/api/agent/*` endpoints and receives resolved prompt text inside each job.

---

## 1. First deploy (clean machine)

```bash
git clone <repo> artwork-automation
cd artwork-automation

# Create the real .env from the template and fill in the secrets.
cp .env.example .env
# Edit .env and set at least:
#   APP_SECRET_KEY            (random, >=32 chars)
#   APP_USERNAME / APP_PASSWORD_HASH   (python set_password.py)
#   POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB
#   DATABASE_URL  -> postgresql+psycopg://<POSTGRES_USER>:<POSTGRES_PASSWORD>@db:5432/<POSTGRES_DB>
#   (Nextcloud + DECOINKS_SERVICE_SECRET as before, if used)

docker compose up -d --build
```

That single command:
1. starts `db` and waits until `pg_isready` reports healthy,
2. runs `alembic upgrade head` (creates every table — **no manual SQL**),
3. runs `python -m scripts.first_start_import` (idempotent): imports an existing
   `users.json` / `agent_tokens.json` into the DB, seeds prompts, indexes any
   existing `vault/` folders, and backs up the JSON files to `*.imported.<ts>`,
4. starts uvicorn.

Check it came up:
```bash
docker compose ps
docker compose logs -f artwork-automation   # look for "alembic upgrade head" + "first-start import done"
```

### Prompt seeding on first start
Prompts become the DB's own data. On first start the seeder pulls the live text
from Decoinks (if `DECOINKS_SERVICE_SECRET` is set and reachable) and seeds the
`prompts` table, with these rules:
- **Decoinks wins** by default (the manager may have edited prompts there).
- **Config wins** for `AIS.RATIO.PLAN` (CUSTOM_ASPECT_ADVICE, recently rewritten).
- A key present only in one source is taken from that source.
- Seeding is **idempotent**: a key already in the DB is never overwritten, so
  edits made later on the Prompts page are preserved across reboots.

A copy of the Decoinks export is archived under `backups/` so nothing is lost
after the app stops reading Decoinks at runtime.

To preview the per-key comparison **before** anything is written (read-only):
```bash
docker compose run --rm artwork-automation python -m scripts.prompt_seed
```
This prints, per key: `same` / `differs` (with a short diff) / `only_config` /
`only_decoinks`, and which source would win.

### After Decoinks is no longer read at runtime
Prompt **text** now comes from our DB. Decoinks is still contacted for the
run-log (`POST /api/ai/generations`) and the manifest PUT
(`PUT /api/ai/apps/artwork-automation/prompts`, which tells Decoinks which
prompt keys/placeholders this app reads and runs) — those are unchanged, and the
same run data is also recorded locally in the `prompt_runs` table.

---

## 2. Update deploy

```bash
cd artwork-automation
git pull
docker compose up -d --build
```

Migrations run automatically on start (`alembic upgrade head`), so a release
that adds a column/table just applies on the next `up`. The first-start import
is idempotent, so it is safe that it runs again (it no-ops on data already
present).

---

## 3. Daily backup (pg_dump, keep 14 days)

Add a host cron job. It dumps the database from the `db` container to a dated
file and prunes anything older than 14 days.

```bash
sudo mkdir -p /var/backups/artwork
sudo crontab -e
```
Add:
```cron
# 02:30 every day: dump the artwork DB, keep 14 days.
30 2 * * * cd /opt/artwork-automation && /usr/bin/docker compose exec -T db \
  pg_dump -U artwork -d artwork --format=custom \
  > /var/backups/artwork/artwork_$(date +\%F).dump 2>>/var/backups/artwork/backup.log; \
  find /var/backups/artwork -name 'artwork_*.dump' -mtime +14 -delete
```
Adjust `/opt/artwork-automation` to the real path and `-U artwork -d artwork` to
your `POSTGRES_USER` / `POSTGRES_DB`. `--format=custom` dumps are compressed and
restore with `pg_restore`.

Also back up the host bind-mounted files (`vault/`, `output/`, `input/`,
`downloads/`) with your normal file backup — those are NOT in Postgres.

### Restore
```bash
# Recreate an empty database, then restore into it.
docker compose exec -T db dropdb   -U artwork --if-exists artwork
docker compose exec -T db createdb -U artwork artwork
docker compose exec -T db pg_restore -U artwork -d artwork --clean --if-exists \
  < /var/backups/artwork/artwork_2026-01-15.dump
# Restart the app so its in-memory cache reloads from the restored DB.
docker compose restart artwork-automation
```

---

## 4. Rollback (if a release / migration misbehaves)

The database is the only stateful part; roll the **code/image** back and leave
the data in place.

```bash
cd artwork-automation
# 1. Go back to the previous release.
git log --oneline -n 5          # find the previous good commit/tag
git checkout <previous-tag-or-commit>
# 2. Rebuild and start that version.
docker compose up -d --build
```

Notes:
- Migrations are **additive**; a rollback of code does not drop tables. If a
  specific migration must be undone, do it explicitly and deliberately:
  `docker compose run --rm artwork-automation alembic downgrade -1`
  (review the migration's `downgrade()` first — the initial migration drops all
  tables, so never downgrade past `0001_initial` on a live system).
- **Take a `pg_dump` immediately before any risky upgrade** so you can restore
  (section 3) if a migration corrupts data.
- The old `users.json` / `agent_tokens.json` are still present this release (see
  section 6), so a rollback to the pre-DB version still finds its data.

---

## 5. Moving existing `vault/` and `downloads/` into place

These stay as **host bind mounts** (not named volumes), so existing data needs
no move — it is already on the host at `./vault`, `./downloads`, `./input`,
`./output`. On a new host, copy the folders next to `docker-compose.yml` before
`docker compose up`:

```bash
rsync -a /old-host/artwork-automation/vault/      ./vault/
rsync -a /old-host/artwork-automation/downloads/  ./downloads/
rsync -a /old-host/artwork-automation/output/     ./output/
rsync -a /old-host/artwork-automation/input/      ./input/
```

The first start indexes existing `vault/<client>/<job>/run_N/` folders into the
`vault_saves` table automatically (idempotent).

> If you ever want vault/downloads on a **named volume** instead of a bind
> mount, create the volume, `docker cp` the data in, and swap the two lines in
> `docker-compose.yml`. The bind-mount default is kept because it makes host
> file backups and inspection trivial.

---

## 6. When to remove the `users.json` / `agent_tokens.json` bind mounts

For **one release** the compose file still bind-mounts `users.json` and
`agent_tokens.json` so a rollback to the pre-DB version keeps working. After
first start these are imported into the DB and backed up to
`*.imported.<timestamp>` (the originals are left in place; the DB is the source
of truth and the stale JSON is no longer read).

Once you are confident you will not roll back (typically the release after this
one), remove these two lines from `docker-compose.yml`:

```yaml
      - ./agent_tokens.json:/app/agent_tokens.json
      - ./users.json:/app/users.json
```
then `docker compose up -d`. Keep the `*.imported.*` backups somewhere safe.

---

## 7. Local development

**Does local dev need Postgres? — Yes, if you want persistence.**

The app now talks to a database. It will still *boot* without one (it degrades
gracefully: prompts fall back to the built-in text in `config/workflows.py`,
and job write-throughs are deferred and retried), but **jobs, Previous Runs,
prompt edits and users will not persist** until a DB is reachable. So for any
real local work, run a Postgres.

The easiest local Postgres is the Compose `db` service on its own:
```bash
# From the repo root, with .env filled in:
docker compose up -d db                      # just the database
export DATABASE_URL=postgresql+psycopg://artwork:artwork@localhost:5432/artwork
# (publish the db port first if you want host access — add "5432:5432" under
#  the db service's ports, or run your own local Postgres and point at it.)
alembic upgrade head
python -m scripts.first_start_import         # seed prompts / import any json
uvicorn app:app --reload --port 8000
```

Or point `DATABASE_URL` at any Postgres you already run locally:
```bash
export DATABASE_URL=postgresql+psycopg://<user>:<pass>@localhost:5432/<db>
alembic upgrade head
uvicorn app:app --reload --port 8000
```

Notes for local dev:
- The container is pinned to Python 3.11 and `SQLAlchemy==2.0.36`. On a newer
  local interpreter (3.13/3.14) that exact SQLAlchemy may not import — use a
  3.11 venv to match the container, or a current 2.0.x locally for convenience.
- With **no** `DECOINKS_SERVICE_SECRET` set, prompt seeding uses the config
  built-ins (logged clearly). With a secret set but Decoinks unreachable, the
  seeder intentionally seeds **nothing** and retries next start (so it can never
  mask the manager's Decoinks edits); the app serves built-in text meanwhile.
- SQLite is **not** a supported substitute — the schema uses Postgres types
  (JSONB, BIGSERIAL). Use Postgres locally.

---

## 8. Scaling note (future)

The job-claim uses an in-process lock, which is correct for the single app
container here. If you ever run **multiple** app replicas against one DB, switch
the claim in `agent_next_job` to a DB-level `SELECT ... FOR UPDATE SKIP LOCKED`
on the oldest `queued` row so two replicas can't hand the same job to two agents.
