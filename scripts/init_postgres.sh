#!/bin/bash
set -e

# Create the airflow metadata database
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    SELECT 'CREATE DATABASE airflow'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec
    GRANT ALL PRIVILEGES ON DATABASE airflow TO "$POSTGRES_USER";
EOSQL

# Create application tables in the health database
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE TABLE IF NOT EXISTS health_logs (
        id                 SERIAL PRIMARY KEY,
        log_date           DATE UNIQUE NOT NULL,
        weather            VARCHAR(20),
        temperature        INTEGER,
        num_meals          INTEGER,
        num_junkfood       INTEGER,          -- must be <= num_meals
        meal_size_lunch    INTEGER,          -- 1 (small) to 5 (big)
        meal_size_dinner   INTEGER,          -- 1 (small) to 5 (big)
        sugar              INTEGER,          -- 1 (very low) to 10 (very high)
        stomach_issues     INTEGER,          -- 1 (none) to 10 (a lot)
        sleep_quality      INTEGER,          -- 1 (very bad) to 10 (very good)
        exercise_gym       BOOLEAN,
        exercise_morning   BOOLEAN,
        stress             INTEGER,          -- 1 (none) to 10 (very stressed)
        social_activity    INTEGER,          -- 1 (isolated) to 10 (very social)
        social_conflicts   INTEGER,          -- 1 (none) to 10 (a lot)
        tiredness_score    INTEGER,          -- 1 (very tired) to 10 (not tired at all)
        created_at         TIMESTAMP DEFAULT NOW(),
        updated_at         TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id                  SERIAL PRIMARY KEY,
        log_date            DATE UNIQUE NOT NULL,
        pred_random_forest  FLOAT,
        pred_gradient_boost FLOAT,
        pred_ridge          FLOAT,
        pred_extra_trees    FLOAT,
        pred_ensemble       FLOAT NOT NULL,
        model_version       VARCHAR(50),
        created_at          TIMESTAMP DEFAULT NOW()
    );

    -- Append-only history: every inference run adds a row, nothing is overwritten.
    -- This lets you trace how each model's prediction for a given date evolved
    -- as more training data was added over time.
    CREATE TABLE IF NOT EXISTS prediction_history (
        id                  SERIAL PRIMARY KEY,
        log_date            DATE NOT NULL,
        pred_random_forest  FLOAT,
        pred_gradient_boost FLOAT,
        pred_ridge          FLOAT,
        pred_extra_trees    FLOAT,
        pred_ensemble       FLOAT NOT NULL,
        model_version       VARCHAR(50),
        run_at              TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_ph_log_date ON prediction_history(log_date);
EOSQL

echo "Postgres initialization complete."
