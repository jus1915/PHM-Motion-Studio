-- =============================================================================
-- PostgreSQL initialisation script
-- Runs once when the postgres container is first created.
--
-- The default database 'mlflow' (owned by devuser) is created automatically
-- by the POSTGRES_DB env var in docker-compose.yml.
-- This script creates the additional 'airflow' database that Airflow needs.
-- =============================================================================

-- Create the Airflow metadata database.
CREATE DATABASE airflow;

-- Grant full access to the shared service user.
GRANT ALL PRIVILEGES ON DATABASE airflow TO devuser;
