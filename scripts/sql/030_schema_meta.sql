-- NL2SQL semantic metadata schema (ported from legacy MySQL ``meta.sql``).
--
-- The subagent that translates natural language into SQL against the ``dw``
-- schema reads this metadata for schema linking. Empty in Phase-1; populated
-- by a registration script in Phase-2 alongside the subagent.

CREATE SCHEMA IF NOT EXISTS meta;
SET search_path = meta, public;


DROP TABLE IF EXISTS meta.table_info CASCADE;
CREATE TABLE meta.table_info (
    id          VARCHAR(64) PRIMARY KEY,
    name        VARCHAR(128),
    role        VARCHAR(32),
    description TEXT
);
COMMENT ON COLUMN meta.table_info.id IS '表编号';
COMMENT ON COLUMN meta.table_info.name IS '表名称';
COMMENT ON COLUMN meta.table_info.role IS '表类型(fact/dim)';
COMMENT ON COLUMN meta.table_info.description IS '表描述';


DROP TABLE IF EXISTS meta.column_info CASCADE;
CREATE TABLE meta.column_info (
    id          VARCHAR(64) PRIMARY KEY,
    name        VARCHAR(128),
    type        VARCHAR(64),
    role        VARCHAR(32),
    examples    JSONB,
    description TEXT,
    alias       JSONB,
    table_id    VARCHAR(64)
);
COMMENT ON COLUMN meta.column_info.id IS '列编号';
COMMENT ON COLUMN meta.column_info.name IS '列名称';
COMMENT ON COLUMN meta.column_info.type IS '数据类型';
COMMENT ON COLUMN meta.column_info.role IS '列类型(primary_key,foreign_key,measure,dimension)';
COMMENT ON COLUMN meta.column_info.examples IS '数据示例';
COMMENT ON COLUMN meta.column_info.description IS '列描述';
COMMENT ON COLUMN meta.column_info.alias IS '列别名';
COMMENT ON COLUMN meta.column_info.table_id IS '所属表编号';


DROP TABLE IF EXISTS meta.metric_info CASCADE;
CREATE TABLE meta.metric_info (
    id               VARCHAR(64) PRIMARY KEY,
    name             VARCHAR(128),
    description      TEXT,
    relevant_columns JSONB,
    alias            JSONB
);
COMMENT ON COLUMN meta.metric_info.id IS '指标编码';
COMMENT ON COLUMN meta.metric_info.name IS '指标名称';
COMMENT ON COLUMN meta.metric_info.description IS '指标描述';
COMMENT ON COLUMN meta.metric_info.relevant_columns IS '关联的列';
COMMENT ON COLUMN meta.metric_info.alias IS '指标别名';


DROP TABLE IF EXISTS meta.column_metric CASCADE;
CREATE TABLE meta.column_metric (
    column_id VARCHAR(64),
    metric_id VARCHAR(64),
    PRIMARY KEY (column_id, metric_id)
);
COMMENT ON COLUMN meta.column_metric.column_id IS '列编号';
COMMENT ON COLUMN meta.column_metric.metric_id IS '指标编号';
