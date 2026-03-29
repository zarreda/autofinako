"""Retrieve earnings call transcripts from Nuvolos (Snowflake).

Supports two connection modes:

1. **Inside Nuvolos** — ``connect_nuvolos()`` with username + SF token (token auth)
2. **Outside Nuvolos** — ``connect_nuvolos()`` with username only.
   When no token is provided and no RSA key is configured, falls back
   to ``externalbrowser`` SSO which opens your browser for 2FA.

All connection parameters come from environment variables (never hardcoded):
- ``NUVOLOS_USERNAME`` — Snowflake username
- ``NUVOLOS_SF_TOKEN`` — Snowflake access token (optional)
- ``SNOWFLAKE_RSA_KEY`` — Path to RSA private key file (optional)
- ``SNOWFLAKE_ACCOUNT`` — Snowflake account identifier
- ``DB_NAME`` — Snowflake database name
- ``SCHEMA_NAME`` — Snowflake schema name
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DB_NAME = "essec_metalab/ako_earnings_prediction"
_DEFAULT_SCHEMA_NAME = "master/development"
_DEFAULT_SNOWFLAKE_ACCOUNT = "alphacruncher.eu-central-1"


@dataclass
class EarningsCall:
    """Container for a single earnings call transcript."""

    company_name: str
    year: str
    quarter: str
    company_id: str
    transcript_id: str
    transcript: str


# ------------------------------------------------------------------
# Connection
# ------------------------------------------------------------------


def connect_nuvolos(
    username: str | None = None,
    sf_token: str | None = None,
    db_name: str | None = None,
    schema_name: str | None = None,
) -> Any:
    """Connect to the Nuvolos Snowflake database.

    All parameters fall back to environment variables if not provided.
    """
    import snowflake.connector

    username = username or os.environ.get("NUVOLOS_USERNAME", "")
    sf_token = sf_token or os.environ.get("NUVOLOS_SF_TOKEN", "")
    db_name = db_name or os.environ.get("DB_NAME", _DEFAULT_DB_NAME)
    schema_name = schema_name or os.environ.get("SCHEMA_NAME", _DEFAULT_SCHEMA_NAME)
    account = os.environ.get("SNOWFLAKE_ACCOUNT", _DEFAULT_SNOWFLAKE_ACCOUNT)

    if not username:
        msg = "Username missing. Set NUVOLOS_USERNAME env var or pass it directly."
        raise OSError(msg)

    if sf_token:
        logger.info("Connecting via nuvolos package (token auth) ...")
        os.environ["NUVOLOS_USERNAME"] = username
        os.environ["NUVOLOS_SF_TOKEN"] = sf_token
        try:
            from nuvolos import get_raw_connection

            return get_raw_connection(
                username=username,
                password=sf_token,
                dbname=db_name,
                schemaname=schema_name,
            )
        except Exception as e:
            logger.warning("Token auth failed (%s), falling back to browser SSO ...", e)

    rsa_key = os.environ.get("SNOWFLAKE_RSA_KEY", "")
    if rsa_key and os.path.exists(rsa_key):
        logger.info("Connecting via RSA key pair auth ...")
        from nuvolos import get_raw_connection

        return get_raw_connection(
            username=username,
            dbname=db_name,
            schemaname=schema_name,
        )

    logger.info("Connecting via external browser SSO — check your browser for 2FA ...")
    con: Any = snowflake.connector.connect(
        user=username,
        account=account,
        authenticator="externalbrowser",
        database=f'"{db_name}"',
        schema=f'"{schema_name}"',
    )
    logger.info("Connected to Snowflake successfully.")
    return con


# ------------------------------------------------------------------
# Query building and transcript reconstruction
# ------------------------------------------------------------------


def _build_query(
    company_name: str,
    year: str,
    quarter: str,
    db_name: str,
    schema_name: str,
) -> str:
    """Build the SQL query for a single earnings call."""
    safe_name = company_name.replace("'", "''")
    schema = f'"{db_name}"."{schema_name}"'
    return f"""
    SELECT
        e.keyDevId,
        rel.documentid,
        rel.objectid,
        rel.documentobjectreltypeid,
        comp.companyId,
        t.TRANSCRIPTCREATIONDATEUTC,
        t.transcriptid,
        t.transcriptPresentationTypeId,
        t.TRANSCRIPTCOLLECTIONTYPEID,
        e.headline,
        tc.componentText,
        tc.transcriptComponentTypeId,
        tc.componentOrder,
        ecb.fiscalquarter,
        ecb.fiscalyear
    FROM {schema}.ciqTranscript t (NOLOCK)
    JOIN {schema}.ciqTranscriptComponent tc (NOLOCK)
        ON tc.transcriptid = t.transcriptId
    JOIN {schema}.ciqEventPit e (NOLOCK)
        ON e.keyDevId = t.keyDevId
    JOIN {schema}.ciqEventToObjectToEventTypePit ete (NOLOCK)
        ON ete.keyDevId = t.keyDevId
    JOIN {schema}.ciqCompany comp (NOLOCK)
        ON comp.companyId = ete.objectId
    JOIN {schema}.ciqEventType et (NOLOCK)
        ON et.keyDevEventTypeId = ete.keyDevEventTypeId
    JOIN {schema}.ciqEventCallBasicInfo ecb (NOLOCK)
        ON ecb.keyDevId = e.keyDevId
    LEFT JOIN {schema}.CIQDOCUMENTOBJECTREL AS rel (NOLOCK)
        ON rel.objectid = e.keyDevId
    WHERE 1=1
        AND et.keyDevEventTypeId = '48'
        AND t.transcriptPresentationTypeId = '5'
        AND comp.companyname = '{safe_name}'
        AND ecb.fiscalyear = {year}
        AND ecb.fiscalquarter = {quarter}
    """


def _reconstruct_transcript(df: pd.DataFrame) -> pd.DataFrame:
    """Apply priority filtering and reconstruct the transcript text.

    Priority: Audited (8) > Edited (2) > Proofed (1).
    Keeps Main content (2) and Q&A (4) sections only.
    """
    for priority in (8, 2, 1):
        if priority in df["transcriptcollectiontypeid"].unique():
            df = df.loc[df["transcriptcollectiontypeid"] == priority]
            break

    df = df[df["documentobjectreltypeid"].isnull() | (df["documentobjectreltypeid"] == "14")]

    df = df.loc[df["transcriptcreationdateutc"] == df["transcriptcreationdateutc"].max()]

    df = df.drop_duplicates(subset="componenttext")
    df = df.loc[df["transcriptcomponenttypeid"].isin([2, 4])]
    df = df.sort_values(by="componentorder", ascending=True)

    return df


def _run_query(query: str, con: Any) -> pd.DataFrame:
    """Execute a SQL query, handling both raw DBAPI and SQLAlchemy connections."""
    try:
        df: pd.DataFrame = pd.read_sql(query, con=con)
    except Exception:
        cursor = con.cursor()
        try:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=columns)
        finally:
            cursor.close()
    df.columns = [c.lower() for c in df.columns]
    return df


# ------------------------------------------------------------------
# Public retrieval API
# ------------------------------------------------------------------


def get_transcript(
    company_name: str,
    year: str,
    quarter: str,
    con: Any,
) -> EarningsCall:
    """Retrieve a single earnings call transcript."""
    db_name = os.environ.get("DB_NAME", _DEFAULT_DB_NAME)
    schema_name = os.environ.get("SCHEMA_NAME", _DEFAULT_SCHEMA_NAME)

    query = _build_query(company_name, year, quarter, db_name, schema_name)
    df = _run_query(query, con)

    if df.empty:
        msg = f"No transcript found for {company_name} {year} Q{quarter}"
        raise ValueError(msg)

    df = _reconstruct_transcript(df)

    full_text = "\n".join(df["componenttext"].astype(str))

    raw_cid = df.iloc[0]["companyid"]
    company_id = str(raw_cid) if raw_cid is not None else "0"
    raw_tid = df.iloc[0]["transcriptid"]
    transcript_id = str(raw_tid) if raw_tid is not None else "0"

    logger.info(
        "Retrieved transcript: %s %s Q%s (%d chars)",
        company_name,
        year,
        quarter,
        len(full_text),
    )

    return EarningsCall(
        company_name=company_name,
        year=year,
        quarter=quarter,
        company_id=company_id,
        transcript_id=transcript_id,
        transcript=full_text,
    )


def get_transcripts(
    earnings_call_list: list[tuple[str, str, str]],
    con: Any,
) -> list[EarningsCall]:
    """Retrieve multiple earnings call transcripts."""
    results: list[EarningsCall] = []
    for company_name, year, quarter in earnings_call_list:
        try:
            ec = get_transcript(company_name, year, quarter, con)
            results.append(ec)
        except ValueError:
            logger.warning(
                "No transcript found for %s %s Q%s",
                company_name,
                year,
                quarter,
            )
    return results
