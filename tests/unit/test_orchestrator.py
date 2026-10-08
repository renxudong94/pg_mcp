"""Unit tests for QueryOrchestrator.

This module tests the orchestrator's coordination of the query pipeline,
including retry logic, error handling, and integration with all components.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    LLMError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    QueryRequest,
    ResultValidationResult,
    ReturnType,
)
from pg_mcp.models.schema import ColumnInfo, DatabaseSchema, TableInfo
from pg_mcp.resilience.circuit_breaker import CircuitBreaker, CircuitState
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator


@pytest.fixture
def mock_schema() -> DatabaseSchema:
    """Create a reusable mock database schema."""
    return DatabaseSchema(
        database_name="test_db",
        tables=[
            TableInfo(
                schema_name="public",
                table_name="users",
                columns=[
                    ColumnInfo(
                        name="id",
                        data_type="integer",
                        is_nullable=False,
                        is_primary_key=True,
                    ),
                ],
            )
        ],
        version="15.0",
    )


class TestDatabaseResolution:
    """Test database name resolution logic."""

    @pytest.fixture
    def mock_pools(self) -> dict[str, MagicMock]:
        """Create mock connection pools."""
        return {
            "db1": MagicMock(),
            "db2": MagicMock(),
        }

    @pytest.fixture
    def orchestrator(self, mock_pools: dict[str, MagicMock]) -> QueryOrchestrator:
        """Create orchestrator with mocked components."""
        return QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools=mock_pools,
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

    def test_resolve_database_specified_valid(self, orchestrator: QueryOrchestrator) -> None:
        """Test resolving a specified valid database."""
        result = orchestrator._resolve_database("db1")
        assert result == "db1"

    def test_resolve_database_specified_invalid(self, orchestrator: QueryOrchestrator) -> None:
        """Test resolving a specified but invalid database."""
        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database("nonexistent")

        assert "not found" in str(exc_info.value).lower()
        assert "db1" in exc_info.value.details["available_databases"]
        assert "db2" in exc_info.value.details["available_databases"]

    def test_resolve_database_auto_select_single(self) -> None:
        """Test auto-selecting when only one database available."""
        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"only_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        result = orchestrator._resolve_database(None)
        assert result == "only_db"

    def test_resolve_database_auto_select_multiple_fails(
        self, orchestrator: QueryOrchestrator
    ) -> None:
        """Test that auto-select fails when multiple databases available."""
        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database(None)

        assert "multiple databases" in str(exc_info.value).lower()
        assert "db1" in exc_info.value.details["available_databases"]

    def test_resolve_database_no_databases(self) -> None:
        """Test error when no databases configured."""
        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database(None)

        assert "no databases configured" in str(exc_info.value).lower()


class TestSQLGenerationWithRetry:
    """Test SQL generation with retry logic."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(
            database_name="test_db",
            tables=[
                TableInfo(
                    schema_name="public",
                    table_name="users",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="integer",
                            is_nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnInfo(
                            name="name",
                            data_type="varchar(255)",
                            is_nullable=False,
                        ),
                    ],
                )
            ],
            version="15.0",
        )

    @pytest.mark.asyncio
    async def test_generate_sql_success_first_attempt(self, mock_schema: DatabaseSchema) -> None:
        """Test successful SQL generation on first attempt."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None  # No exception = valid

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=3),
            validation_config=ValidationConfig(),
        )

        # Execute
        sql, validation_result, _tokens = await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=mock_schema,
            request_id="test-123",
        )

        # Verify
        assert sql == "SELECT * FROM users;"
        assert validation_result.is_valid is True
        assert validation_result.is_select is True
        mock_generator.generate.assert_called_once()
        mock_validator.validate_or_raise.assert_called_once_with("SELECT * FROM users;")

    @pytest.mark.asyncio
    async def test_generate_sql_retry_on_validation_failure(
        self, mock_schema: DatabaseSchema
    ) -> None:
        """Test retry logic when validation fails."""
        # Setup mocks - first attempt fails validation, second succeeds
        mock_generator = AsyncMock()
        mock_generator.generate.side_effect = [
            "SELECT * FROM user;",  # First attempt (wrong table name)
            "SELECT * FROM users;",  # Second attempt (correct)
        ]

        mock_validator = MagicMock()
        # First call raises error, second call succeeds
        mock_validator.validate_or_raise.side_effect = [
            SQLParseError('relation "user" does not exist'),
            None,  # Success on second attempt
        ]

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=3),
            validation_config=ValidationConfig(),
        )

        # Execute
        sql, validation_result, _tokens = await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=mock_schema,
            request_id="test-123",
        )

        # Verify
        assert sql == "SELECT * FROM users;"
        assert validation_result.is_valid is True
        assert mock_generator.generate.call_count == 2
        assert mock_validator.validate_or_raise.call_count == 2

        # Verify retry included error feedback
        second_call = mock_generator.generate.call_args_list[1]
        assert second_call.kwargs["previous_attempt"] == "SELECT * FROM user;"
        assert 'relation "user" does not exist' in second_call.kwargs["error_feedback"]

    @pytest.mark.asyncio
    async def test_generate_sql_fails_after_max_retries(self, mock_schema: DatabaseSchema) -> None:
        """Test failure after exhausting all retries."""
        # Setup mocks - all attempts fail validation
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "DELETE FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.side_effect = SecurityViolationError(
            "DELETE statements are not allowed"
        )

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=2),
            validation_config=ValidationConfig(),
        )

        # Execute and verify exception
        with pytest.raises(SecurityViolationError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Delete all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "DELETE statements are not allowed" in str(exc_info.value)
        # Should attempt max_retries + 1 times (initial + retries)
        assert mock_generator.generate.call_count == 3
        assert orchestrator.circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_generate_sql_circuit_breaker_open(self, mock_schema: DatabaseSchema) -> None:
        """Test that open circuit breaker prevents SQL generation."""
        orchestrator = QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(circuit_breaker_threshold=1),
            validation_config=ValidationConfig(),
        )

        # Manually open the circuit breaker
        orchestrator.circuit_breaker._state = CircuitState.OPEN
        orchestrator.circuit_breaker._failure_count = 5

        # Attempt should fail immediately
        with pytest.raises(LLMError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "temporarily unavailable" in str(exc_info.value).lower()
        assert "circuit breaker" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_generate_sql_unexpected_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of unexpected errors during generation."""
        mock_generator = AsyncMock()
        mock_generator.generate.side_effect = RuntimeError("Unexpected error")

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=1),
            validation_config=ValidationConfig(),
        )

        with pytest.raises(LLMError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "unexpectedly" in str(exc_info.value).lower()
        assert orchestrator.circuit_breaker.failure_count == 1


class TestResultValidation:
    """Test result validation logic."""

    @pytest.mark.asyncio
    async def test_validate_results_success(self) -> None:
        """Test successful result validation."""
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = ResultValidationResult(
            confidence=85,
            explanation="Results match the question well",
            suggestion=None,
            is_acceptable=True,
        )

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=mock_validator,
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True),
        )

        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 85
        mock_validator.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_results_disabled(self) -> None:
        """Test that validation is skipped when disabled."""
        mock_validator = AsyncMock()

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=mock_validator,
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
        )

        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 100
        mock_validator.validate.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_results_failure_does_not_raise(self) -> None:
        """Test that validation failures don't raise exceptions."""
        mock_validator = AsyncMock()
        mock_validator.validate.side_effect = Exception("Validation failed")

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=mock_validator,
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True),
        )

        # Should not raise, returns default confidence
        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 100


class TestExecuteQueryFlow:
    """Test complete query execution flow."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(
            database_name="test_db",
            tables=[
                TableInfo(
                    schema_name="public",
                    table_name="users",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="integer",
                            is_nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnInfo(
                            name="name",
                            data_type="varchar(255)",
                            is_nullable=False,
                        ),
                    ],
                )
            ],
            version="15.0",
        )

    @pytest.mark.asyncio
    async def test_execute_query_sql_only(self, mock_schema: DatabaseSchema) -> None:
        """Test executing query with return_type=SQL."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        assert response.generated_sql == "SELECT * FROM users;"
        assert response.validation is not None
        assert response.validation.is_valid is True
        assert response.data is None  # No execution for SQL-only
        assert response.error is None

    @pytest.mark.asyncio
    async def test_execute_query_with_results(self, mock_schema: DatabaseSchema) -> None:
        """Test executing query with return_type=RESULT."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT id, name FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_executor = AsyncMock()
        mock_executor.execute.return_value = (
            [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
            2,  # total count
        )

        mock_result_validator = AsyncMock()
        mock_result_validator.validate.return_value = ResultValidationResult(
            confidence=90,
            explanation="Good results",
            suggestion=None,
            is_acceptable=True,
        )

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=mock_executor,
            result_validator=mock_result_validator,
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        assert response.generated_sql == "SELECT id, name FROM users;"
        assert response.data is not None
        assert response.data.row_count == 2
        assert len(response.data.rows) == 2
        assert response.data.columns == ["id", "name"]
        assert response.confidence == 90
        assert response.error is None

    @pytest.mark.asyncio
    async def test_execute_query_schema_not_cached(self) -> None:
        """Test loading schema when not in cache."""
        mock_schema = DatabaseSchema(
            database_name="test_db",
            tables=[],
            version="15.0",
        )

        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Not in cache
        mock_cache.load = AsyncMock(return_value=mock_schema)

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_pool = MagicMock()

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": mock_pool},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Test query",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify schema was loaded
        mock_cache.load.assert_called_once_with("test_db", mock_pool)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_execute_query_schema_load_fails(self) -> None:
        """Test handling of schema load failure."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.load = AsyncMock(side_effect=Exception("DB connection failed"))

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Test query",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "schema" in response.error.message.lower()
        assert response.generated_sql is None

    @pytest.mark.asyncio
    async def test_execute_query_validation_error(self) -> None:
        """Test handling of SQL validation errors."""
        mock_schema = DatabaseSchema(
            database_name="test_db",
            tables=[],
            version="15.0",
        )

        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "DELETE FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.side_effect = SecurityViolationError("DELETE not allowed")

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=1),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Delete all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "DELETE not allowed" in response.error.message
        assert response.error.code == "security_violation"

    @pytest.mark.asyncio
    async def test_execute_query_execution_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of SQL execution errors."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_executor = AsyncMock()
        mock_executor.execute.side_effect = DatabaseError("Query execution failed")

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=mock_executor,
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "execution failed" in response.error.message.lower()
        assert response.error.code == "database_error"

    @pytest.mark.asyncio
    async def test_execute_query_unexpected_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of unexpected errors."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.side_effect = RuntimeError("Unexpected error")

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert response.error.code == "internal_error"
        assert "internal server error" in response.error.message.lower()

    @pytest.mark.asyncio
    async def test_execute_query_auto_select_database(self, mock_schema: DatabaseSchema) -> None:
        """Test auto-selecting database when only one available."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"only_db": MagicMock()},  # Only one database
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute without specifying database
        request = QueryRequest(
            question="Test query",
            database=None,  # No database specified
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        # Verify schema was fetched for auto-selected database
        mock_cache.get.assert_called_once_with("only_db")


class TestMultiDatabaseExecutorDispatch:
    """Test that requests are dispatched to the executor of the resolved DB."""

    def _build_orchestrator(
        self,
        schema: DatabaseSchema,
        executors: dict[str, AsyncMock],
        pools: dict[str, MagicMock],
        **kwargs: object,
    ) -> QueryOrchestrator:
        """Build an orchestrator wired with per-database executors."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_result_validator = AsyncMock()
        mock_result_validator.validate.return_value = ResultValidationResult(
            confidence=80,
            explanation="ok",
            suggestion=None,
            is_acceptable=True,
        )

        return QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            result_validator=mock_result_validator,
            schema_cache=mock_cache,
            pools=pools,
            sql_executors=executors,  # type: ignore[arg-type]
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
            **kwargs,  # type: ignore[arg-type]
        )

    @pytest.mark.asyncio
    async def test_dispatches_to_executor_of_resolved_database(
        self, mock_schema: DatabaseSchema
    ) -> None:
        """Each database uses its own executor, never another database's."""
        executor_a = AsyncMock()
        executor_a.execute.return_value = ([{"a": 1}], 1)
        executor_b = AsyncMock()
        executor_b.execute.return_value = ([{"b": 2}], 1)

        orchestrator = self._build_orchestrator(
            schema=mock_schema,
            executors={"db_a": executor_a, "db_b": executor_b},
            pools={"db_a": MagicMock(), "db_b": MagicMock()},
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db_b", return_type=ReturnType.RESULT)
        )

        assert response.success is True
        executor_b.execute.assert_called_once_with("SELECT 1;")
        executor_a.execute.assert_not_called()

    def test_missing_executor_is_rejected(self, mock_schema: DatabaseSchema) -> None:
        """Requesting a database without an executor must raise, not fall back."""
        orchestrator = self._build_orchestrator(
            schema=mock_schema,
            executors={"db_a": AsyncMock()},
            pools={"db_a": MagicMock(), "db_b": MagicMock()},
        )

        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._get_executor("db_b")

        assert "no sql executor" in str(exc_info.value).lower()
        assert exc_info.value.details["database"] == "db_b"

    def test_requires_an_executor(self) -> None:
        """Constructing without any executor is a programming error."""
        with pytest.raises(ValueError, match="sql_executors"):
            QueryOrchestrator(
                sql_generator=MagicMock(),
                sql_validator=MagicMock(),
                result_validator=MagicMock(),
                schema_cache=MagicMock(),
                pools={"db": MagicMock()},
                resilience_config=ResilienceConfig(),
                validation_config=ValidationConfig(),
            )


class TestQuestionValidation:
    """Test question-length enforcement from ValidationConfig."""

    @pytest.mark.asyncio
    async def test_question_exceeding_limit_returns_validation_error(
        self, mock_schema: DatabaseSchema
    ) -> None:
        """Questions longer than max_question_length are rejected."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        orchestrator = QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(max_question_length=10),
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="x" * 11, database="test_db", return_type=ReturnType.SQL)
        )

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "validation_failed"
        assert response.error.details is not None
        assert response.error.details["max_length"] == 10


class TestRateLimiterIntegration:
    """Test that the query rate limiter gates request processing."""

    @pytest.mark.asyncio
    async def test_query_rate_limit_rejection_returns_error(
        self, mock_schema: DatabaseSchema
    ) -> None:
        """When no query slot is free, the request fails with rate_limit_exceeded."""
        limiter = MultiRateLimiter(query_limit=1, llm_limit=1)
        # Occupy the only slot so the orchestrator cannot acquire one.
        await limiter.query_limiter.acquire()

        try:
            mock_cache = MagicMock()
            mock_cache.get.return_value = mock_schema

            orchestrator = QueryOrchestrator(
                sql_generator=AsyncMock(),
                sql_validator=MagicMock(),
                sql_executor=MagicMock(),
                result_validator=MagicMock(),
                schema_cache=mock_cache,
                pools={"test_db": MagicMock()},
                resilience_config=ResilienceConfig(rate_limit_timeout=0.1),
                validation_config=ValidationConfig(),
                rate_limiter=limiter,
            )

            response = await orchestrator.execute_query(
                QueryRequest(question="q", database="test_db", return_type=ReturnType.SQL)
            )

            assert response.success is False
            assert response.error is not None
            assert response.error.code == "rate_limit_exceeded"
        finally:
            limiter.query_limiter.release()
            await asyncio.sleep(0)


class TestMetricsIntegration:
    """Test that metrics are recorded for every request."""

    def _build_orchestrator(
        self, schema: DatabaseSchema, generator: AsyncMock
    ) -> tuple[QueryOrchestrator, MagicMock]:
        """Build an orchestrator with a mock metrics collector."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = schema

        metrics = MagicMock()
        orchestrator = QueryOrchestrator(
            sql_generator=generator,
            sql_validator=MagicMock(validate_or_raise=MagicMock(return_value=None)),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
            metrics=metrics,
        )
        return orchestrator, metrics

    @pytest.mark.asyncio
    async def test_success_records_success_metrics(self, mock_schema: DatabaseSchema) -> None:
        """A successful request increments the success counter and timers."""
        generator = AsyncMock()
        generator.generate.return_value = "SELECT 1;"
        orchestrator, metrics = self._build_orchestrator(mock_schema, generator)

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="test_db", return_type=ReturnType.SQL)
        )

        assert response.success is True
        metrics.increment_query_request.assert_called_once_with("success", "test_db")
        metrics.query_duration.observe.assert_called_once()
        metrics.increment_llm_call.assert_called_once_with("generate_sql")
        metrics.observe_llm_latency.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_records_error_metrics(self, mock_schema: DatabaseSchema) -> None:
        """A rejected request increments the error counter and rejection metric."""
        generator = AsyncMock()
        generator.generate.return_value = "DELETE FROM users;"

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema
        metrics = MagicMock()
        validator = MagicMock()
        validator.validate_or_raise.side_effect = SecurityViolationError("DELETE not allowed")

        orchestrator = QueryOrchestrator(
            sql_generator=generator,
            sql_validator=validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=0),
            validation_config=ValidationConfig(),
            metrics=metrics,
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="test_db", return_type=ReturnType.SQL)
        )

        assert response.success is False
        metrics.increment_query_request.assert_called_once_with("error", "test_db")
        metrics.increment_sql_rejected.assert_called_once_with("SecurityViolationError")


class TestSharedCircuitBreaker:
    """Test that the injected circuit breaker is used and honoured."""

    @pytest.mark.asyncio
    async def test_injected_circuit_breaker_is_shared(self, mock_schema: DatabaseSchema) -> None:
        """The orchestrator keeps the provided circuit breaker instance."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)

        orchestrator = QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
            circuit_breaker=breaker,
        )

        assert orchestrator.circuit_breaker is breaker

        # Opening the shared breaker must block generation immediately.
        breaker._state = CircuitState.OPEN
        breaker._failure_count = 5

        with pytest.raises(LLMError):
            await orchestrator._generate_sql_with_retry(
                question="q",
                schema=mock_schema,
                request_id="test-123",
            )


class TestRetryBackoff:
    """Test exponential backoff between SQL generation retries."""

    @pytest.mark.asyncio
    async def test_backoff_sleeps_between_retries(self, mock_schema: DatabaseSchema) -> None:
        """Retries wait ``retry_delay * backoff_factor**attempt`` seconds."""
        generator = AsyncMock()
        generator.generate.side_effect = ["SELECT * FROM user;", "SELECT * FROM users;"]

        validator = MagicMock()
        validator.validate_or_raise.side_effect = [
            SQLParseError("no such table"),
            None,
        ]

        orchestrator = QueryOrchestrator(
            sql_generator=generator,
            sql_validator=validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=2, retry_delay=0.1, backoff_factor=2.0),
            validation_config=ValidationConfig(),
        )

        with patch(
            "pg_mcp.services.orchestrator.asyncio.sleep", new_callable=AsyncMock
        ) as sleep_mock:
            sql, _validation, _tokens = await orchestrator._generate_sql_with_retry(
                question="q",
                schema=mock_schema,
                request_id="test-123",
            )

        assert sql == "SELECT * FROM users;"
        assert sleep_mock.await_count == 1
        # First retry delay should equal retry_delay * backoff_factor**0
        assert sleep_mock.await_args_list[0].args[0] == pytest.approx(0.1)
