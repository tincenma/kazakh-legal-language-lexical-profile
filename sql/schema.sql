-- Normalised schema for KazakhLawCorpus-clean (SQLite)
-- acts N:1 bodies, acts N:M areas (via act_areas), acts N:M forms (via act_forms), acts N:1 sections

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS act_forms;
DROP TABLE IF EXISTS act_areas;
DROP TABLE IF EXISTS acts;
DROP TABLE IF EXISTS forms;
DROP TABLE IF EXISTS areas;
DROP TABLE IF EXISTS sections;
DROP TABLE IF EXISTS bodies;

CREATE TABLE bodies (
    body_id   INTEGER PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE
);

CREATE TABLE sections (
    section_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE
);

CREATE TABLE areas (
    area_id   INTEGER PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE
);

CREATE TABLE forms (
    form_id   INTEGER PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE
);

CREATE TABLE acts (
    act_id        INTEGER PRIMARY KEY,
    law_id        TEXT NOT NULL UNIQUE,
    title         TEXT,
    date_adopted  TEXT,            -- ISO YYYY-MM-DD
    year          INTEGER,
    decade        INTEGER,
    is_repealed   INTEGER,         -- 0 / 1
    legal_force   TEXT,
    region        TEXT,
    body_id       INTEGER REFERENCES bodies(body_id),
    section_id    INTEGER REFERENCES sections(section_id),
    n_tokens      INTEGER          -- tokens in title + description
);

CREATE TABLE act_areas (
    act_id   INTEGER NOT NULL REFERENCES acts(act_id),
    area_id  INTEGER NOT NULL REFERENCES areas(area_id),
    PRIMARY KEY (act_id, area_id)
);

CREATE TABLE act_forms (
    act_id   INTEGER NOT NULL REFERENCES acts(act_id),
    form_id  INTEGER NOT NULL REFERENCES forms(form_id),
    PRIMARY KEY (act_id, form_id)
);

CREATE INDEX idx_acts_year    ON acts(year);
CREATE INDEX idx_acts_body    ON acts(body_id);
CREATE INDEX idx_acts_section ON acts(section_id);
CREATE INDEX idx_act_areas_area ON act_areas(area_id);
