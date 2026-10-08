"""Unit tests for the observability modules (logging, metrics, tracing).

These tests verify the building blocks that are wired into the query pipeline:
structured logging with sensitive-data redaction, Prometheus metric helpers and
request-id context propagation.
"""

import json
import logging

import pytest

from pg_mcp.observability.logging import (
    JSONFormatter,
    SensitiveDataFilter,
    TextFormatter,
    configure_logging,
    get_logger,
)
from pg_mcp.observability.metrics import metrics
from pg_mcp.observability.tracing import (
    TraceContext,
    clear_request_id,
    generate_request_id,
    get_request_id,
    get_tracing_logger,
    request_context,
    set_request_id,
    trace_async,
    trace_sync,
)


class TestRequestContext:
    """Tests for request-id generation and propagation."""

    def test_generate_request_id_is_unique(self) -> None:
        """Generated IDs are unique UUID strings."""
        first = generate_request_id()
        second = generate_request_id()
        assert first != second
        assert len(first) == 36

    def test_set_get_clear_request_id(self) -> None:
        """The request id can be set, read and cleared."""
        set_request_id("rid-1")
        try:
            assert get_request_id() == "rid-1"
        finally:
            clear_request_id()

        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_request_context_restores_previous_id(self) -> None:
        """The context manager restores the previous value on exit."""
        set_request_id("outer")
        try:
            async with request_context() as generated:
                assert generated != "outer"
                assert get_request_id() == generated
            assert get_request_id() == "outer"
        finally:
            clear_request_id()

    @pytest.mark.asyncio
    async def test_request_context_uses_provided_id(self) -> None:
        """An explicit id is used verbatim."""
        async with request_context("fixed-id") as rid:
            assert rid == "fixed-id"
            assert get_request_id() == "fixed-id"

    def test_trace_context_model(self) -> None:
        """TraceContext carries the tracing metadata."""
        ctx = TraceContext(request_id="rid", operation="query", metadata={"k": "v"})
        assert ctx.request_id == "rid"
        assert ctx.parent_id is None
        assert ctx.metadata == {"k": "v"}


class TestTracingDecorators:
    """Tests for the trace_async / trace_sync decorators."""

    @pytest.mark.asyncio
    async def test_trace_async_adds_request_id_to_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Log records produced inside the call carry the request id."""

        @trace_async(operation="work")
        async def work() -> str:
            logging.getLogger("trace.async").info("working")
            return "ok"

        with caplog.at_level(logging.INFO):
            async with request_context("rid-async"):
                assert await work() == "ok"

        assert any(getattr(r, "request_id", None) == "rid-async" for r in caplog.records)

    @pytest.mark.asyncio
    async def test_trace_async_without_context(self) -> None:
        """Without a request context the function still runs."""

        @trace_async()
        async def work(value: int) -> int:
            return value + 1

        clear_request_id()
        assert await work(1) == 2

    def test_trace_sync_adds_request_id_to_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """The sync decorator propagates the request id too."""

        @trace_sync(operation="sync_work")
        def work() -> str:
            logging.getLogger("trace.sync").warning("syncing")
            return "done"

        with caplog.at_level(logging.WARNING):
            set_request_id("rid-sync")
            try:
                assert work() == "done"
            finally:
                clear_request_id()

        assert any(getattr(r, "request_id", None) == "rid-sync" for r in caplog.records)

    def test_trace_sync_without_context(self) -> None:
        """Without a request context the sync function still runs."""

        @trace_sync()
        def work(value: int) -> int:
            return value * 2

        clear_request_id()
        assert work(3) == 6


class TestTracingLogger:
    """Tests for the TracingLogger wrapper."""

    def test_includes_request_id_in_extra(self, caplog: pytest.LogCaptureFixture) -> None:
        """The logger injects the current request id."""
        logger = get_tracing_logger("tracing_logger_test")
        assert isinstance(logger, type(get_tracing_logger("x")))

        with caplog.at_level(logging.DEBUG):
            set_request_id("rid-logger")
            try:
                logger.debug("d")
                logger.info("i")
                logger.warning("w")
                logger.error("e")
                logger.critical("c")
            finally:
                clear_request_id()

        ids = [getattr(r, "request_id", None) for r in caplog.records]
        assert ids.count("rid-logger") >= 5

    def test_exception_logs_traceback(self, caplog: pytest.LogCaptureFixture) -> None:
        """exception() records the traceback."""
        logger = get_tracing_logger("tracing_exc_test")
        with caplog.at_level(logging.ERROR):
            try:
                raise ValueError("boom")
            except ValueError:
                logger.exception("failed")

        assert any(r.exc_info is not None for r in caplog.records)

    def test_no_request_id_when_absent(self, caplog: pytest.LogCaptureFixture) -> None:
        """No request_id is injected when there is no active context."""
        clear_request_id()
        logger = get_tracing_logger("tracing_none_test")
        with caplog.at_level(logging.INFO):
            logger.info("plain")

        assert all(getattr(r, "request_id", None) is None for r in caplog.records)


class TestSensitiveDataFilter:
    """Tests for redaction of sensitive values."""

    def _make_record(self) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        return record

    def test_redacts_top_level_sensitive_keys(self) -> None:
        """Sensitive attribute names are replaced with a placeholder."""
        record = self._make_record()
        record.password = "super-secret"  # type: ignore[attr-defined]
        record.api_key = "sk-123"  # type: ignore[attr-defined]

        assert SensitiveDataFilter().filter(record) is True

        assert record.password == "***REDACTED***"
        assert record.api_key == "***REDACTED***"

    def test_redacts_nested_dict_values(self) -> None:
        """Sensitive keys inside nested dicts are redacted recursively."""
        record = self._make_record()
        record.context = {"token": "abc", "nested": {"secret": "xyz", "ok": 1}}  # type: ignore[attr-defined]

        SensitiveDataFilter().filter(record)

        assert record.context["token"] == "***REDACTED***"
        assert record.context["nested"]["secret"] == "***REDACTED***"
        assert record.context["nested"]["ok"] == 1

    def test_sanitizes_args(self) -> None:
        """Positional log args are sanitized in place."""
        record = self._make_record()
        record.args = {"password": "p", "user": "u"}

        SensitiveDataFilter().filter(record)

        assert record.args["password"] == "***REDACTED***"
        assert record.args["user"] == "u"


class TestFormatters:
    """Tests for the JSON and text log formatters."""

    def _make_record(self, with_request_id: bool = True) -> logging.LogRecord:
        record = logging.LogRecord(
            name="pg_mcp.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="processed %s",
            args=("query",),
            exc_info=None,
        )
        if with_request_id:
            record.request_id = "rid-fmt"  # type: ignore[attr-defined]
        record.database = "mydb"  # type: ignore[attr-defined]
        return record

    def test_json_formatter_outputs_json(self) -> None:
        """JSONFormatter produces a parseable JSON object."""
        formatted = JSONFormatter().format(self._make_record())
        payload = json.loads(formatted)

        assert payload["level"] == "INFO"
        assert payload["message"] == "processed query"
        assert payload["request_id"] == "rid-fmt"
        assert payload["extra"]["database"] == "mydb"

    def test_json_formatter_without_request_id(self) -> None:
        """request_id is omitted when not set."""
        payload = json.loads(JSONFormatter().format(self._make_record(with_request_id=False)))
        assert "request_id" not in payload

    def test_json_formatter_with_exception(self) -> None:
        """Exceptions are rendered into the payload."""
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            import sys

            record = self._make_record()
            record.exc_info = sys.exc_info()

        payload = json.loads(JSONFormatter().format(record))
        assert "RuntimeError" in payload["exception"]

    def test_text_formatter_readable(self) -> None:
        """TextFormatter renders a human-readable line."""
        formatted = TextFormatter().format(self._make_record())
        assert "[INFO]" in formatted
        assert "processed query" in formatted
        assert "request_id=rid-fmt" in formatted

    def test_text_formatter_with_exception(self) -> None:
        """TextFormatter appends the traceback."""
        try:
            raise ValueError("nope")
        except ValueError:
            import sys

            record = self._make_record()
            record.exc_info = sys.exc_info()

        formatted = TextFormatter().format(record)
        assert "ValueError" in formatted


class TestConfigureLogging:
    """Tests for configure_logging."""

    def _restore(self, handlers: list[logging.Handler], level: int) -> None:
        root = logging.getLogger()
        root.handlers[:] = handlers
        root.setLevel(level)

    def test_configure_json_logging(self) -> None:
        """JSON logging installs a single JSON handler at the right level."""
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            configure_logging(level="DEBUG", log_format="json", enable_sensitive_filter=True)
            assert root.level == logging.DEBUG
            assert len(root.handlers) == 1
            assert isinstance(root.handlers[0].formatter, JSONFormatter)
            assert any(isinstance(f, SensitiveDataFilter) for f in root.handlers[0].filters)
        finally:
            self._restore(handlers, level)

    def test_configure_text_logging(self) -> None:
        """Text logging installs a TextFormatter handler."""
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            configure_logging(level="INFO", log_format="text", enable_sensitive_filter=False)
            assert isinstance(root.handlers[0].formatter, TextFormatter)
            assert not root.handlers[0].filters
        finally:
            self._restore(handlers, level)

    def test_get_logger_returns_named_logger(self) -> None:
        """get_logger returns a standard logger for the module name."""
        logger = get_logger("pg_mcp.custom")
        assert logger.name == "pg_mcp.custom"


class TestMetricsHelpers:
    """Tests for the Prometheus metric helper methods."""

    def test_increment_query_request(self) -> None:
        """Query request counter increments per (status, database)."""
        label = {"status": "success", "database": "metrics_test"}
        before = metrics.query_requests.labels(**label)._value.get()
        metrics.increment_query_request(**label)
        assert metrics.query_requests.labels(**label)._value.get() == before + 1

    def test_increment_llm_call_and_tokens(self) -> None:
        """LLM call and token counters increment."""
        op = {"operation": "generate_sql_test"}
        before_calls = metrics.llm_calls.labels(**op)._value.get()
        before_tokens = metrics.llm_tokens_used.labels(**op)._value.get()

        metrics.increment_llm_call(**op)
        metrics.increment_llm_tokens(operation=op["operation"], tokens=7)

        assert metrics.llm_calls.labels(**op)._value.get() == before_calls + 1
        assert metrics.llm_tokens_used.labels(**op)._value.get() == before_tokens + 7

    def test_observe_latencies(self) -> None:
        """Latency histograms record observations."""
        metrics.observe_llm_latency("latency_test", 0.5)
        metrics.observe_db_query_duration(0.25)

        assert metrics.llm_latency.labels(operation="latency_test")._sum.get() >= 0.5
        assert metrics.db_query_duration._sum.get() >= 0.25

    def test_increment_sql_rejected(self) -> None:
        """SQL rejection counter increments per reason."""
        before = metrics.sql_rejected.labels(reason="unit_test")._value.get()
        metrics.increment_sql_rejected("unit_test")
        assert metrics.sql_rejected.labels(reason="unit_test")._value.get() == before + 1

    def test_gauges(self) -> None:
        """Gauges can be set for connections and schema cache age."""
        metrics.set_db_connections_active("gauge_test", 3)
        metrics.set_schema_cache_age("gauge_test", 12.5)

        assert metrics.db_connections_active.labels(database="gauge_test")._value.get() == 3
        assert metrics.schema_cache_age.labels(database="gauge_test")._value.get() == 12.5
