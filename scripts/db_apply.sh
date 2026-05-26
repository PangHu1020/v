#!/usr/bin/env bash
# Apply Postgres schema migrations in numeric order.
#
# Reads ``POSTGRES_DSN`` from the environment (or .env if loaded). Each .sql
# file under scripts/sql/ runs once on a fresh database; idempotent guards
# (CREATE SCHEMA IF NOT EXISTS, DROP TABLE IF EXISTS CASCADE) make re-runs
# safe during development. Production migrations are explicit and auditable.

set -euo pipefail

DSN="${POSTGRES_DSN:-postgresql://postgres:postgres@localhost:5432/agent}"
SQL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/sql && pwd)"

echo "Applying SQL migrations from ${SQL_DIR}"
echo "Target: ${DSN}"

for f in "${SQL_DIR}"/*.sql; do
    echo "  -> ${f##*/}"
    psql "${DSN}" -v ON_ERROR_STOP=1 -f "${f}"
done

echo "All migrations applied."
