-- Analytical queries. Each query starts with "-- name: <id>" and "-- description: <text>".
-- src/database.py parses this file and loads each result into a pandas DataFrame.

-- name: acts_per_decade
-- description: Number of acts and share of repealed acts per decade (GROUP BY)
SELECT decade,
       COUNT(*)                                   AS n_acts,
       ROUND(100.0 * SUM(is_repealed) / COUNT(*), 1) AS repealed_pct
FROM acts
WHERE decade IS NOT NULL
GROUP BY decade
ORDER BY decade;

-- name: top_bodies
-- description: Most productive adopting bodies with more than 500 acts (JOIN + GROUP BY/HAVING)
SELECT b.name AS body, COUNT(*) AS n_acts, MIN(a.year) AS first_year, MAX(a.year) AS last_year
FROM acts a
JOIN bodies b ON b.body_id = a.body_id
GROUP BY b.body_id
HAVING COUNT(*) > 500
ORDER BY n_acts DESC
LIMIT 20;

-- name: top_areas
-- description: Most frequent legal relation areas through the many-to-many link (JOIN x2 + GROUP BY)
SELECT ar.name AS area, COUNT(DISTINCT aa.act_id) AS n_acts
FROM act_areas aa
JOIN areas ar ON ar.area_id = aa.area_id
JOIN acts a   ON a.act_id  = aa.act_id
GROUP BY ar.area_id
ORDER BY n_acts DESC
LIMIT 20;

-- name: multi_area_acts
-- description: Acts that belong to more areas than average (subquery in HAVING)
SELECT a.law_id, a.title, COUNT(*) AS n_areas
FROM acts a
JOIN act_areas aa ON aa.act_id = a.act_id
GROUP BY a.act_id
HAVING COUNT(*) > (SELECT AVG(cnt) * 3 FROM (SELECT COUNT(*) AS cnt FROM act_areas GROUP BY act_id))
ORDER BY n_areas DESC
LIMIT 20;

-- name: section_growth
-- description: Acts per section and decade with change vs previous decade (CTE + LAG window)
WITH per_decade AS (
    SELECT s.name AS section, a.decade, COUNT(*) AS n_acts
    FROM acts a
    JOIN sections s ON s.section_id = a.section_id
    WHERE a.decade IS NOT NULL
    GROUP BY s.name, a.decade
)
SELECT section, decade, n_acts,
       LAG(n_acts) OVER (PARTITION BY section ORDER BY decade)            AS prev_decade,
       n_acts - LAG(n_acts) OVER (PARTITION BY section ORDER BY decade)   AS change
FROM per_decade
ORDER BY section, decade;

-- name: body_rank_per_decade
-- description: Top 3 adopting bodies in every decade (CTE + RANK window)
WITH counts AS (
    SELECT a.decade, b.name AS body, COUNT(*) AS n_acts
    FROM acts a
    JOIN bodies b ON b.body_id = a.body_id
    WHERE a.decade IS NOT NULL
    GROUP BY a.decade, b.name
), ranked AS (
    SELECT decade, body, n_acts,
           RANK() OVER (PARTITION BY decade ORDER BY n_acts DESC) AS rnk
    FROM counts
)
SELECT decade, rnk, body, n_acts
FROM ranked
WHERE rnk <= 3
ORDER BY decade, rnk;

-- name: cumulative_acts
-- description: Running total of adopted acts by year (SUM window)
SELECT year, COUNT(*) AS n_acts,
       SUM(COUNT(*)) OVER (ORDER BY year ROWS UNBOUNDED PRECEDING) AS cumulative
FROM acts
WHERE year IS NOT NULL
GROUP BY year
ORDER BY year;

-- name: area_share_by_decade
-- description: Share of each top-10 area within its decade (CTE + subquery + window SUM)
WITH top_areas AS (
    SELECT area_id FROM act_areas GROUP BY area_id ORDER BY COUNT(*) DESC LIMIT 10
), counts AS (
    SELECT a.decade, ar.name AS area, COUNT(*) AS n
    FROM act_areas aa
    JOIN acts a   ON a.act_id = aa.act_id
    JOIN areas ar ON ar.area_id = aa.area_id
    WHERE aa.area_id IN (SELECT area_id FROM top_areas) AND a.decade IS NOT NULL
    GROUP BY a.decade, ar.name
)
SELECT decade, area, n,
       ROUND(100.0 * n / SUM(n) OVER (PARTITION BY decade), 1) AS share_pct
FROM counts
ORDER BY decade, share_pct DESC;

-- name: text_length_by_section
-- description: Average text length per section and its deviation from the overall mean (JOIN + scalar subquery)
SELECT s.name AS section, COUNT(*) AS n_acts,
       ROUND(AVG(a.n_tokens), 1) AS avg_tokens,
       ROUND(AVG(a.n_tokens) - (SELECT AVG(n_tokens) FROM acts), 1) AS diff_from_mean
FROM acts a
JOIN sections s ON s.section_id = a.section_id
GROUP BY s.section_id
ORDER BY avg_tokens DESC;

-- name: repealed_by_form
-- description: Repeal rate per act form, forms with at least 100 acts (JOIN + HAVING + window percent rank)
SELECT f.name AS act_form, COUNT(*) AS n_acts,
       ROUND(100.0 * SUM(a.is_repealed) / COUNT(*), 1) AS repealed_pct,
       ROUND(PERCENT_RANK() OVER (ORDER BY 1.0 * SUM(a.is_repealed) / COUNT(*)), 2) AS pct_rank
FROM act_forms af
JOIN forms f ON f.form_id = af.form_id
JOIN acts a  ON a.act_id = af.act_id
GROUP BY f.form_id
HAVING COUNT(*) >= 100
ORDER BY repealed_pct DESC;
