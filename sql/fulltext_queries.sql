-- Queries on the full-text sample (menu 10). Same format as queries.sql:
-- each query starts with "-- name: <id>" and "-- description: <text>".

-- name: fulltext_length_by_section_decade
-- description: Full-text length vs title + description length per section and decade (CTE + JOIN + GROUP BY/HAVING)
WITH sample AS (
    SELECT a.decade, s.name AS section, a.n_tokens AS title_tokens, f.n_tokens AS full_tokens
    FROM fulltexts f
    JOIN acts a     ON a.act_id = f.act_id
    JOIN sections s ON s.section_id = a.section_id
)
SELECT section, decade, COUNT(*) AS n_acts,
       ROUND(AVG(title_tokens), 1) AS avg_title_tokens,
       ROUND(AVG(full_tokens), 0)  AS avg_full_tokens,
       ROUND(AVG(1.0 * full_tokens / MAX(title_tokens, 1)), 1) AS avg_length_ratio
FROM sample
GROUP BY section, decade
HAVING COUNT(*) >= 5
ORDER BY section, decade;

-- name: longest_fulltext_per_decade
-- description: The three longest full texts in every decade (window ROW_NUMBER over a JOIN)
SELECT decade, rank_in_decade, law_id, full_tokens, title
FROM (
    SELECT a.decade, a.law_id, a.title, f.n_tokens AS full_tokens,
           ROW_NUMBER() OVER (PARTITION BY a.decade ORDER BY f.n_tokens DESC) AS rank_in_decade
    FROM fulltexts f
    JOIN acts a ON a.act_id = f.act_id
)
WHERE rank_in_decade <= 3
ORDER BY decade, rank_in_decade;

-- name: stemming_effect_by_section
-- description: How much stemming shrinks each act's vocabulary, by section, ranked (subquery + window RANK)
SELECT section, n_acts, avg_reduction_pct,
       RANK() OVER (ORDER BY avg_reduction_pct DESC) AS rank_by_reduction
FROM (
    SELECT s.name AS section, COUNT(*) AS n_acts,
           ROUND(AVG(100.0 * (f.n_content_types - f.n_stem_types) / MAX(f.n_content_types, 1)), 1)
               AS avg_reduction_pct
    FROM fulltexts f
    JOIN acts a     ON a.act_id = f.act_id
    JOIN sections s ON s.section_id = a.section_id
    GROUP BY s.section_id
)
ORDER BY rank_by_reduction;
