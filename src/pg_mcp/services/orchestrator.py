"""Query orchestrator for coordinating the complete query flow.

This module provides the QueryOrchestrator class that coordinates all components
of the query processing pipeline: SQL generation, validation, execution, and result
validation. It implements retry logic with exponential backoff, request tracing,
rate limiting, metrics collection and comprehensive error handling.

The orchestrator depends on abstractions (generator, validator, executor,
rate limiter, metrics) but never instantiates them, keeping the responsibility of
wiring concrete implementations in the composition root (the MCP server).
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    LLMError,
    PgMcpError,
    RateLimitExceededError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
    ValidationError,
)
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.tracing import (
    clear_request_id,
    generate_request_id,
    set_request_id,
)
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

if TYPE_CHECKING:
    from pg_mcp.models.schema import DatabaseSchema
    from pg_mcp.observability.metrics import MetricsCollector

logger = logging.getLogger(__name__)

# Upper bound for a single retry backoff sleep (seconds)
_MAX_BACKOFF_SECONDS = 30.0


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    This class coordinates SQL generation, validation, execution, and result
    validation. It implements retry logic with exponential backoff, circuit
    breaker protection for the LLM, concurrency rate limiting and metrics
    collection.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"mydb": pool},
        ...     sql_executors={"mydb": executor},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="mydb"
        ... ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        result_validator: ResultValidator,
        schema_cache: SchemaCache,
        pools: dict[str, Pool],
        resilience_config: ResilienceConfig,
        validation_config: ValidationConfig,
        sql_executors: dict[str, SQLExecutor] | None = None,
        sql_executor: SQLExecutor | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        rate_limiter: MultiRateLimiter | None = None,
        metrics: "MetricsCollector | None" = None,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: SQL validation service.
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries and circuit breaker.
            validation_config: Validation configuration including thresholds.
            sql_executors: Mapping of database name to executor. When provided,
                each request is dispatched to the executor of the resolved
                database, preventing cross-database access.
            sql_executor: Deprecated single executor used as a fallback for every
                database. Prefer ``sql_executors`` so requests cannot reach the
                wrong database.
            circuit_breaker: Shared circuit breaker. A new one is created from
                ``resilience_config`` when omitted.
            rate_limiter: Optional concurrency limiter for queries and LLM calls.
            metrics: Optional Prometheus metrics collector.

        Raises:
            ValueError: If neither ``sql_executors`` nor ``sql_executor`` is given.
        """
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.result_validator = result_validator
        self.schema_cache = schema_cache
        self.pools = pools
        self.resilience_config = resilience_config
        self.validation_config = validation_config
        self.rate_limiter = rate_limiter
        self.metrics = metrics

        # Prefer the per-database mapping; fall back to a single default executor.
        if sql_executors:
            self.sql_executors: dict[str, SQLExecutor] = dict(sql_executors)
            self._default_executor: SQLExecutor | None = None
        elif sql_executor is not None:
            self.sql_executors = {}
            self._default_executor = sql_executor
        else:
            raise ValueError("Either 'sql_executors' or 'sql_executor' must be provided")

        # Use the shared circuit breaker when supplied so its state is global.
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=resilience_config.circuit_breaker_threshold,
            recovery_timeout=resilience_config.circuit_breaker_timeout,
        )

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute complete query flow from question to results.

        This method orchestrates the entire pipeline while applying
        cross-cutting concerns:

        1. Generate a request ID and propagate it via the tracing context
        2. Resolve and validate the target database
        3. Acquire a query rate-limiter slot
        4. Load the schema, generate/validate SQL (with retries) and execute
        5. Record metrics and translate failures into structured errors

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.

        Example:
            >>> response = await orchestrator.execute_query(
            ...     QueryRequest(question="Count all users", return_type="result")
            ... )
            >>> if response.success:
            ...     print(f"Found {response.data.row_count} rows")
        """
        request_id = generate_request_id()
        set_request_id(request_id)
        started_at = time.perf_counter()
        database_name = request.database or "default"

        logger.info(
            "Starting query execution",
            extra={"request_id": request_id, "question": request.question[:100]},
        )

        try:
            # Resolve the database up-front so metrics/logs carry the real name.
            database_name = self._resolve_database(request.database)

            if self.rate_limiter is not None:
                async with self.rate_limiter.for_queries(
                    timeout=self.resilience_config.rate_limit_timeout
                ):
                    response = await self._execute_pipeline(request, request_id, database_name)
            else:
                response = await self._execute_pipeline(request, request_id, database_name)
        except PgMcpError as e:
            # Handle known application errors
            logger.warning(
                "Query execution failed with known error",
                extra={
                    "request_id": request_id,
                    "error_code": e.code,
                    "error_message": str(e),
                },
            )
            response = self._error_response(e.code, e.message, e.details)
        except TimeoutError:
            logger.warning(
                "Query rejected by rate limiter",
                extra={"request_id": request_id},
            )
            response = self._error_response(
                ErrorCode.RATE_LIMIT_EXCEEDED,
                "Too many concurrent queries, please retry later",
                {"database": database_name},
            )
        except Exception as e:
            # Handle unexpected errors
            logger.exception(
                "Query execution failed with unexpected error",
                extra={"request_id": request_id},
            )
            response = self._error_response(
                ErrorCode.INTERNAL_ERROR,
                f"Internal server error: {e!s}",
                {"error_type": type(e).__name__},
            )
        finally:
            clear_request_id()

        status = "success" if response.success else "error"
        duration_seconds = time.perf_counter() - started_at

        if self.metrics is not None:
            self.metrics.query_duration.observe(duration_seconds)
            self.metrics.increment_query_request(status, database_name)

        return response

    async def _execute_pipeline(
        self,
        request: QueryRequest,
        request_id: str,
        database_name: str,
    ) -> QueryResponse:
        """Run the query pipeline, raising ``PgMcpError`` on failure.

        Args:
            request: Query request containing question and parameters.
            request_id: Request ID for tracking.
            database_name: Already-resolved target database name.

        Returns:
            QueryResponse: Successful response (SQL-only or with results).

        Raises:
            PgMcpError: Any known failure while processing the query.
        """
        # Step 0: enforce configured question length limit
        self._validate_question(request.question)

        logger.debug(
            "Resolved database",
            extra={"request_id": request_id, "database": database_name},
        )

        # Step 1: Get schema from cache (loading it on demand)
        schema = await self._load_schema(database_name, request_id)

        # Step 2: Generate and validate SQL with retry logic
        generated_sql, validation_result, tokens_used = await self._generate_sql_with_retry(
            question=request.question,
            schema=schema,
            request_id=request_id,
            database_name=database_name,
        )

        # Step 3: If return_type is SQL, return early
        if request.return_type == ReturnType.SQL:
            logger.info(
                "Returning SQL only",
                extra={"request_id": request_id, "sql_length": len(generated_sql)},
            )
            return QueryResponse(
                success=True,
                generated_sql=generated_sql,
                validation=validation_result,
                data=None,
                error=None,
                confidence=100,
                tokens_used=tokens_used,
            )

        # Step 4: Execute SQL using the executor of the resolved database
        executor = self._get_executor(database_name)
        logger.debug("Executing SQL", extra={"request_id": request_id})

        start_time = time.perf_counter()
        results, total_count = await executor.execute(generated_sql)
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        if self.metrics is not None:
            self.metrics.observe_db_query_duration(execution_time_ms / 1000)

        logger.info(
            "SQL executed successfully",
            extra={
                "request_id": request_id,
                "row_count": total_count,
                "execution_time_ms": execution_time_ms,
            },
        )

        # Step 5: Validate results (non-blocking, failures don't fail the request)
        result_confidence = await self._validate_results_safely(
            question=request.question,
            sql=generated_sql,
            results=results,
            row_count=total_count,
            request_id=request_id,
        )

        if result_confidence < self.validation_config.min_confidence_score:
            logger.warning(
                "Result confidence below configured minimum",
                extra={
                    "request_id": request_id,
                    "confidence": result_confidence,
                    "min_confidence": self.validation_config.min_confidence_score,
                },
            )

        # Step 6: Build successful response
        query_result = QueryResult(
            columns=list(results[0].keys()) if results else [],
            rows=results,
            row_count=len(results),  # Limited row count (after max_rows applied)
            execution_time_ms=execution_time_ms,
        )

        return QueryResponse(
            success=True,
            generated_sql=generated_sql,
            validation=validation_result,
            data=query_result,
            error=None,
            confidence=result_confidence,
            tokens_used=tokens_used,
        )

    def _validate_question(self, question: str) -> None:
        """Validate the question against the configured maximum length.

        Args:
            question: The user's natural language question.

        Raises:
            ValidationError: If the question exceeds ``max_question_length``.
        """
        max_length = self.validation_config.max_question_length
        if len(question) > max_length:
            raise ValidationError(
                message=f"Question exceeds maximum length of {max_length} characters",
                details={"length": len(question), "max_length": max_length},
            )

    async def _load_schema(self, database_name: str, request_id: str) -> "DatabaseSchema":
        """Return the schema for ``database_name``, loading it when not cached.

        Args:
            database_name: Target database name.
            request_id: Request ID for tracking.

        Returns:
            DatabaseSchema: The database schema.

        Raises:
            DatabaseError: If no connection pool exists for the database.
            SchemaLoadError: If schema introspection fails.
        """
        schema = self.schema_cache.get(database_name)
        if schema is None:
            pool = self.pools.get(database_name)
            if pool is None:
                raise DatabaseError(
                    message=f"No connection pool available for database '{database_name}'",
                    details={"database": database_name},
                )
            try:
                schema = await self.schema_cache.load(database_name, pool)
            except Exception as e:
                raise SchemaLoadError(
                    message=f"Failed to load schema for database '{database_name}': {e!s}",
                    details={"database": database_name, "error": str(e)},
                ) from e

        if self.metrics is not None:
            age = self.schema_cache.get_cache_age(database_name)
            if age is not None:
                self.metrics.set_schema_cache_age(database_name, age)

        logger.debug(
            "Schema loaded",
            extra={
                "request_id": request_id,
                "database": database_name,
                "tables": len(schema.tables),
            },
        )
        return schema

    def _get_executor(self, database_name: str) -> SQLExecutor:
        """Select the executor responsible for ``database_name``.

        Args:
            database_name: Resolved database name.

        Returns:
            SQLExecutor: The executor bound to the database.

        Raises:
            DatabaseError: If no executor is configured for the database.
        """
        executor = self.sql_executors.get(database_name)
        if executor is not None:
            return executor
        if self._default_executor is not None:
            return self._default_executor
        raise DatabaseError(
            message=f"No SQL executor configured for database '{database_name}'",
            details={
                "database": database_name,
                "available_databases": sorted(self.sql_executors),
            },
        )

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request or auto-select.

        If database is specified, validate it exists.
        If not specified and only one database available, auto-select it.

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If database is invalid or cannot be auto-selected.

        Example:
            >>> name = orchestrator._resolve_database("mydb")  # Validates "mydb" exists
            >>> name = orchestrator._resolve_database(None)  # Auto-selects if only one DB
        """
        if database is not None:
            # Validate specified database exists
            if database not in self.pools:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": list(self.pools.keys()),
                    },
                )
            return database

        # Auto-select if only one database available
        available_dbs = list(self.pools.keys())
        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        # Multiple databases, must specify
        raise DatabaseError(
            message="Multiple databases available, please specify which to query",
            details={"available_databases": available_dbs},
        )

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: Any,
        request_id: str,
        database_name: str = "default",
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry logic on validation failures.

        This method implements a retry loop that:
        1. Checks circuit breaker state
        2. Generates SQL using LLM (rate limited, metrics recorded)
        3. Validates the generated SQL
        4. On validation failure, waits with exponential backoff and retries
           with error feedback
        5. Records success/failure to circuit breaker

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            request_id: Request ID for tracking.
            database_name: Target database name (for logging).

        Returns:
            tuple: (generated_sql, validation_result, tokens_used)

        Raises:
            LLMError: If circuit breaker is open or generation fails.
            SecurityViolationError: If SQL fails validation after all retries.
            SQLParseError: If SQL cannot be parsed.
            RateLimitExceededError: If the LLM rate limiter rejects the call.
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message="SQL generation service is temporarily unavailable (circuit breaker open)",
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        tokens_used: int | None = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "Generating SQL",
                    extra={
                        "request_id": request_id,
                        "database": database_name,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                    },
                )

                # Generate SQL (rate limited + metrics)
                generated_sql = await self._call_llm(
                    question=question,
                    schema=schema,
                    previous_attempt=previous_sql,
                    error_feedback=error_feedback,
                    request_id=request_id,
                )

                logger.debug(
                    "SQL generated",
                    extra={
                        "request_id": request_id,
                        "sql_length": len(generated_sql),
                    },
                )

                # Validate SQL
                try:
                    self.sql_validator.validate_or_raise(generated_sql)
                except (SecurityViolationError, SQLParseError) as validation_error:
                    if self.metrics is not None:
                        self.metrics.increment_sql_rejected(type(validation_error).__name__)

                    if attempt < max_retries:
                        # Record as failure and retry with feedback
                        logger.warning(
                            "SQL validation failed, retrying with feedback",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)
                        await self._backoff(attempt, request_id)
                        continue
                    else:
                        # Out of retries, record failure and raise
                        self.circuit_breaker.record_failure()
                        logger.error(
                            "SQL validation failed after all retries",
                            extra={
                                "request_id": request_id,
                                "attempts": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        raise

                # Validation successful
                self.circuit_breaker.record_success()
                logger.info(
                    "SQL generated and validated successfully",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                    },
                )

                # Build validation result
                validation_result = ValidationResult(
                    is_valid=True,
                    is_select=True,
                    allows_data_modification=False,
                    uses_blocked_functions=[],
                    error_message=None,
                )

                return generated_sql, validation_result, tokens_used

            except (LLMError, SecurityViolationError, SQLParseError, RateLimitExceededError):
                # Re-raise known errors
                raise
            except Exception as e:
                # Unexpected error during generation
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e!s}",
                    details={"error_type": type(e).__name__},
                ) from e

        # Should not reach here, but just in case
        self.circuit_breaker.record_failure()
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    async def _call_llm(
        self,
        question: str,
        schema: Any,
        previous_attempt: str | None,
        error_feedback: str | None,
        request_id: str,
    ) -> str:
        """Invoke the SQL generator behind the LLM rate limiter.

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            previous_attempt: Previously failed SQL, if any.
            error_feedback: Validation error feedback, if any.
            request_id: Request ID for tracking.

        Returns:
            str: Generated SQL.

        Raises:
            RateLimitExceededError: If no LLM slot becomes available in time.
        """
        llm_started = time.perf_counter()
        try:
            if self.rate_limiter is not None:
                async with self.rate_limiter.for_llm(
                    timeout=self.resilience_config.rate_limit_timeout
                ):
                    generated_sql = await self.sql_generator.generate(
                        question=question,
                        schema=schema,
                        previous_attempt=previous_attempt,
                        error_feedback=error_feedback,
                    )
            else:
                generated_sql = await self.sql_generator.generate(
                    question=question,
                    schema=schema,
                    previous_attempt=previous_attempt,
                    error_feedback=error_feedback,
                )
        except TimeoutError as e:
            raise RateLimitExceededError(
                message="LLM rate limit exceeded, please retry later",
                details={"request_id": request_id},
            ) from e

        if self.metrics is not None:
            self.metrics.increment_llm_call("generate_sql")
            self.metrics.observe_llm_latency("generate_sql", time.perf_counter() - llm_started)

        return generated_sql

    async def _backoff(self, attempt: int, request_id: str) -> None:
        """Sleep using exponential backoff before the next retry.

        Args:
            attempt: Zero-based attempt index that just failed.
            request_id: Request ID for tracking.
        """
        delay = min(
            self.resilience_config.retry_delay * (self.resilience_config.backoff_factor**attempt),
            _MAX_BACKOFF_SECONDS,
        )
        logger.info(
            "Backing off before retry",
            extra={"request_id": request_id, "delay_seconds": delay, "attempt": attempt + 1},
        )
        await asyncio.sleep(delay)

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> int:
        """Validate query results with error handling (non-blocking).

        This method attempts to validate results using LLM, but failures
        don't cause the overall query to fail. Returns a confidence score.

        Args:
            question: User's original question.
            sql: Generated SQL query.
            results: Query results.
            row_count: Total row count.
            request_id: Request ID for tracking.

        Returns:
            int: Confidence score (0-100). Returns 100 if validation disabled/fails.

        Example:
            >>> confidence = await orchestrator._validate_results_safely(
            ...     question="Count users",
            ...     sql="SELECT COUNT(*) FROM users",
            ...     results=[{"count": 42}],
            ...     row_count=1,
            ...     request_id="123",
            ... )
        """
        if not self.validation_config.enabled:
            return 100

        try:
            logger.debug(
                "Validating results",
                extra={"request_id": request_id},
            )

            validation_started = time.perf_counter()
            validation_result = await self.result_validator.validate(
                question=question,
                sql=sql,
                results=results,
                row_count=row_count,
            )

            if self.metrics is not None:
                self.metrics.increment_llm_call("validate_result")
                self.metrics.observe_llm_latency(
                    "validate_result", time.perf_counter() - validation_started
                )

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                },
            )

            return validation_result.confidence

        except Exception as e:
            # Log but don't fail the query
            logger.warning(
                "Result validation failed, continuing with default confidence",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
            )
            return 100  # Default to high confidence if validation fails

    @staticmethod
    def _error_response(
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None,
    ) -> QueryResponse:
        """Build a failed ``QueryResponse`` from an error.

        Args:
            code: Error code.
            message: Human-readable error message.
            details: Additional error context.

        Returns:
            QueryResponse: Response with ``success=False``.
        """
        return QueryResponse(
            success=False,
            generated_sql=None,
            validation=None,
            data=None,
            error=ErrorDetail(
                code=code.value,
                message=message,
                details=details,
            ),
            confidence=0,
            tokens_used=None,
        )
