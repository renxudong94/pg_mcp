"""Feature demo for pg-mcp.

Shows the implemented behaviour without needing a real Postgres or OpenAI key:
the schema cache, SQL generator and executors are mocked, everything else
(configuration, validator, orchestrator, rate limiter, metrics) is real code.

Usage:
    PYTHONPATH=src python scripts/demo.py            # all sections
    PYTHONPATH=src python scripts/demo.py 1          # only one section

Sections:
    1   multi-database configuration + table/column/EXPLAIN/write controls
    1b  per-database executor dispatch (no cross-database fallback)
    2   rate limiting + exponential backoff retry
    3   Prometheus output after a single query
    4   response model serialization
"""

import asyncio
import json
import logging
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

from prometheus_client import REGISTRY, generate_latest

from pg_mcp.config.settings import (
    OpenAIConfig,
    ResilienceConfig,
    SecurityConfig,
    Settings,
    ValidationConfig,
)
from pg_mcp.models.errors import SQLParseError
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
)
from pg_mcp.models.schema import ColumnInfo, DatabaseSchema, TableInfo
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.sql_validator import SQLValidator

# The orchestrator and validator log by design; silence them for a clean demo.
logging.getLogger("sqlglot").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

BAR = "=" * 78

TABLES = [
    TableInfo(
        schema_name="public",
        table_name="users",
        columns=[
            ColumnInfo(name="id", data_type="integer", is_nullable=False, is_primary_key=True),
            ColumnInfo(name="name", data_type="text", is_nullable=True),
        ],
    )
]
SCHEMA = DatabaseSchema(database_name="db", tables=TABLES, version="15.0")


def header(title: str) -> None:
    """Print a section header."""
    print("\n" + BAR + "\n" + title + "\n" + BAR)


def check(validator: SQLValidator, sql: str) -> None:
    """Validate one statement and pretty-print the outcome."""
    try:
        validator.validate_or_raise(sql)
        print(f"  [ALLOWED] {sql}")
    except Exception as e:
        print(f"  [BLOCKED] {sql}")
        print(f"            -> {type(e).__name__}: {e}")


def make_orchestrator(**kw: object) -> QueryOrchestrator:
    """Build an orchestrator with mocked generator / cache / executors."""
    gen = AsyncMock()
    gen.generate = AsyncMock(return_value="SELECT id, name FROM users;")

    cache = MagicMock()
    cache.get.return_value = SCHEMA
    cache.get_cache_age.return_value = None

    exec_a = AsyncMock()
    exec_a.execute = AsyncMock(return_value=([{"id": 1}], 1))

    defaults: dict[str, object] = {
        "sql_generator": gen,
        "sql_validator": SQLValidator.from_config(SecurityConfig()),
        "result_validator": AsyncMock(),
        "schema_cache": cache,
        "pools": {"db_a": MagicMock(), "db_b": MagicMock()},
        "sql_executors": {"db_a": exec_a},
        "resilience_config": ResilienceConfig(),
        "validation_config": ValidationConfig(enabled=False),
    }
    defaults.update(kw)
    return QueryOrchestrator(**defaults)  # type: ignore[arg-type]


def demo_security() -> None:
    """Show that config values actually drive the validator."""
    header("1) MULTI-DATABASE + SECURITY CONTROL  (config -> enforced)")

    os.environ["DATABASES"] = '{"analytics": {"name": "analytics", "host": "analytics.host"}}'
    os.environ["SECURITY_BLOCKED_TABLES"] = "secrets, api_keys"
    os.environ["SECURITY_BLOCKED_COLUMNS"] = "users.ssn, password"
    os.environ["SECURITY_ALLOW_EXPLAIN"] = "true"

    settings = Settings(openai=OpenAIConfig(api_key="sk-demo-key-1234"))

    print("configured databases  :", sorted(settings.get_database_configs()))
    print("extra db host         :", settings.get_database_configs()["analytics"].host)
    print("blocked_tables  (env) :", settings.security.blocked_tables)
    print("blocked_columns (env) :", settings.security.blocked_columns)
    print("allow_explain   (env) :", settings.security.allow_explain)

    for key in (
        "DATABASES",
        "SECURITY_BLOCKED_TABLES",
        "SECURITY_BLOCKED_COLUMNS",
        "SECURITY_ALLOW_EXPLAIN",
    ):
        os.environ.pop(key, None)

    print("\n-- restrictive policy (blocked table/column, no EXPLAIN, read-only) --")
    strict = SQLValidator.from_config(
        SecurityConfig(
            blocked_tables=["secrets"],
            blocked_columns=["password"],
            allow_explain=False,
            allow_write_operations=False,
        )
    )
    for sql in (
        "SELECT id FROM users",
        "SELECT * FROM secrets",
        "SELECT id, password FROM users",
        "EXPLAIN SELECT id FROM users",
        "DELETE FROM users WHERE id = 1",
    ):
        check(strict, sql)

    print("\n-- relaxed policy (allow_explain=True, allow_write_operations=True) --")
    relaxed = SQLValidator.from_config(
        SecurityConfig(
            blocked_tables=["secrets"],
            allow_explain=True,
            allow_write_operations=True,
        )
    )
    for sql in (
        "EXPLAIN SELECT id FROM users",
        "UPDATE users SET name = 'x' WHERE id = 1",
        "DROP TABLE users",
    ):
        check(relaxed, sql)


async def demo_dispatch() -> None:
    """Show that each database uses its own executor."""
    header("1b) PER-DATABASE EXECUTOR DISPATCH  (no cross-database fallback)")

    exec_a = AsyncMock()
    exec_a.execute = AsyncMock(return_value=([{"id": 1}], 1))

    orch = make_orchestrator(sql_executors={"db_a": exec_a})

    ok = await orch.execute_query(
        QueryRequest(question="list users", database="db_a", return_type=ReturnType.RESULT)
    )
    print(
        f"  db_a -> success={ok.success} rows={ok.data.row_count} "
        f"executor.execute called={exec_a.execute.await_count}"
    )

    bad = await orch.execute_query(
        QueryRequest(question="list users", database="db_b", return_type=ReturnType.RESULT)
    )
    print(f"  db_b -> success={bad.success} code={bad.error.code}")
    print(f"         message: {bad.error.message}")
    print(f"         details: {bad.error.details}")
    print(f"  db_a executor untouched: {exec_a.execute.await_count} call(s)")


async def demo_resilience() -> None:
    """Show rate-limit rejection and exponential backoff."""
    header("2) RESILIENCE: RATE LIMITING + EXPONENTIAL BACKOFF")

    limiter = MultiRateLimiter(query_limit=1, llm_limit=1)
    await limiter.query_limiter.acquire()  # occupy the only query slot
    try:
        orch = make_orchestrator(
            rate_limiter=limiter,
            resilience_config=ResilienceConfig(rate_limit_timeout=0.2),
        )
        r = await orch.execute_query(QueryRequest(question="q", database="db_a"))
        print(f"  query slot exhausted -> success={r.success} code={r.error.code}")
        print(f"                          message: {r.error.message}")
        print(f"  limiter stats: {limiter.get_all_stats()['queries']}")
    finally:
        limiter.query_limiter.release()
        await asyncio.sleep(0)

    print("\n-- retry with exponential backoff (retry_delay=1.0, factor=2.0) --")
    gen = AsyncMock()
    gen.generate = AsyncMock(
        side_effect=[
            "SELECT * FROM user;",
            "SELECT * FROM userz;",
            "SELECT id FROM users;",
        ]
    )
    validator = MagicMock()
    validator.validate_or_raise = MagicMock(
        side_effect=[
            SQLParseError("unknown table 'user'"),
            SQLParseError("unknown table 'userz'"),
            None,
        ]
    )
    orch = make_orchestrator(
        sql_generator=gen,
        sql_validator=validator,
        resilience_config=ResilienceConfig(max_retries=2, retry_delay=1.0, backoff_factor=2.0),
    )
    with patch("pg_mcp.services.orchestrator.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
        sql, _v, _t = await orch._generate_sql_with_retry(
            question="list users", schema=SCHEMA, request_id="demo-request"
        )
    print("  attempts        : 3 (2 rejections, then success)")
    print("  sleep delays    :", [c.args[0] for c in sleep_mock.await_args_list])
    print("  final SQL       :", sql)


async def demo_metrics() -> None:
    """Show real Prometheus series produced by one query."""
    header("3) OBSERVABILITY: REAL PROMETHEUS OUTPUT AFTER ONE QUERY")

    metrics = MetricsCollector()
    exec_shop = AsyncMock()
    exec_shop.execute = AsyncMock(return_value=([{"id": 1}, {"id": 2}], 2))

    orch = make_orchestrator(
        metrics=metrics,
        pools={"shop": MagicMock()},
        sql_executors={"shop": exec_shop},
    )
    r = await orch.execute_query(
        QueryRequest(question="list all users", database="shop", return_type=ReturnType.RESULT)
    )
    print(
        f"  query: success={r.success} rows={r.data.row_count} "
        f"execution_time_ms={r.data.execution_time_ms:.3f}"
    )

    print("\n-- GET /metrics (pg_mcp_* series) --")
    for line in generate_latest(REGISTRY).decode().splitlines():
        if line.startswith("pg_mcp_") and "_created" not in line:
            print("  " + line)


def demo_model() -> None:
    """Show the single to_dict() implementation and its output."""
    header("4) RESPONSE MODEL: SINGLE to_dict(), NO NULL FIELDS")

    ok = QueryResponse(
        success=True,
        generated_sql="SELECT id, name FROM users",
        validation=None,
        data=QueryResult(columns=["id", "name"], rows=[{"id": 1, "name": "a"}], row_count=1),
        confidence=95,
        tokens_used=137,
    )
    err = QueryResponse(
        success=False,
        error=ErrorDetail(
            code="database_error",
            message="No SQL executor configured for database 'db_b'",
            details={"database": "db_b", "available_databases": ["db_a"]},
        ),
        confidence=0,
    )
    print("success response:")
    print(json.dumps(ok.to_dict(), indent=2, ensure_ascii=False))
    print("\nerror response (no 'data'/'generated_sql' keys, tokens_used always present):")
    print(json.dumps(err.to_dict(), indent=2, ensure_ascii=False))


async def main(section: str) -> None:
    """Run one or all demo sections."""
    if section in ("all", "1"):
        demo_security()
    if section in ("all", "1b"):
        await demo_dispatch()
    if section in ("all", "2"):
        await demo_resilience()
    if section in ("all", "3"):
        await demo_metrics()
    if section in ("all", "4"):
        demo_model()

    if section == "all":
        print("\n" + BAR)
        print("All checks executed against real modules (Postgres / OpenAI mocked only).")
        print(BAR)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "all"))
