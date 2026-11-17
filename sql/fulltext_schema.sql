-- Full texts of the downloaded sample acts (menu 10). acts 1 : 0..1 fulltexts.
-- Loaded by src/database.py: load_fulltexts_to_db(); the rest of the schema is in schema.sql.

DROP TABLE IF EXISTS fulltexts;

CREATE TABLE fulltexts (
    act_id          INTEGER PRIMARY KEY REFERENCES acts(act_id),
    n_chars         INTEGER NOT NULL,
    n_tokens        INTEGER NOT NULL,     -- all Kazakh/Cyrillic word tokens
    n_content_types INTEGER NOT NULL,     -- distinct word forms after stop-word removal
    n_stem_types    INTEGER NOT NULL,     -- distinct suffix stems of those word forms
    text            TEXT NOT NULL         -- cleaned full text
);
