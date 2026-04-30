-- Machine Learning Risk Analytics Work Sample
-- Synthetic SQL examples for EDA, model monitoring, experiment analysis, and executive reporting

-- 1. Exploratory data analysis by customer segment

SELECT
    customer_segment,
    COUNT(DISTINCT case_id) AS total_cases,
    AVG(case_age_days) AS avg_case_age_days,
    AVG(touchpoints) AS avg_touchpoints,
    AVG(resolution_time_hours) AS avg_resolution_time_hours,
    AVG(CAST(sla_breach_flag AS FLOAT)) AS sla_breach_rate,
    AVG(CAST(escalated_flag AS FLOAT)) AS escalation_rate
FROM customer_case_data
GROUP BY customer_segment
ORDER BY escalation_rate DESC;


-- 2. Escalation rate by priority and case type

SELECT
    priority,
    case_type,
    COUNT(DISTINCT case_id) AS total_cases,
    SUM(escalated_flag) AS escalated_cases,
    ROUND(SUM(escalated_flag) * 1.0 / COUNT(DISTINCT case_id), 3) AS escalation_rate
FROM customer_case_data
GROUP BY priority, case_type
ORDER BY escalation_rate DESC;


-- 3. Monthly model monitoring view

SELECT
    case_month,
    COUNT(DISTINCT case_id) AS total_cases,
    ROUND(AVG(CAST(escalated_flag AS FLOAT)), 3) AS actual_escalation_rate,
    ROUND(AVG(predicted_escalation_probability), 3) AS avg_predicted_risk,
    ROUND(AVG(CAST(sla_breach_flag AS FLOAT)), 3) AS sla_breach_rate,
    ROUND(AVG(resolution_time_hours), 1) AS avg_resolution_time_hours
FROM customer_case_scored
GROUP BY case_month
ORDER BY case_month;


-- 4. High-risk intervention list

SELECT
    case_id,
    case_month,
    customer_segment,
    channel,
    case_type,
    priority,
    case_age_days,
    touchpoints,
    sla_breach_flag,
    predicted_escalation_probability,
    recommended_action
FROM customer_case_scored
WHERE predicted_escalation_probability >= 0.70
ORDER BY predicted_escalation_probability DESC, case_age_days DESC;


-- 5. A/B test summary

SELECT
    experiment_group,
    COUNT(DISTINCT case_id) AS total_cases,
    ROUND(AVG(CAST(escalated_flag AS FLOAT)), 3) AS escalation_rate,
    ROUND(AVG(CAST(sla_breach_flag AS FLOAT)), 3) AS sla_breach_rate,
    ROUND(AVG(resolution_time_hours), 1) AS avg_resolution_time_hours
FROM customer_case_data
GROUP BY experiment_group
ORDER BY escalation_rate ASC;


-- 6. Data quality checks for reproducible modeling

SELECT
    COUNT(*) AS total_records,
    SUM(CASE WHEN case_id IS NULL THEN 1 ELSE 0 END) AS missing_case_id,
    SUM(CASE WHEN case_month IS NULL THEN 1 ELSE 0 END) AS missing_case_month,
    SUM(CASE WHEN case_age_days < 0 THEN 1 ELSE 0 END) AS invalid_case_age,
    SUM(CASE WHEN resolution_time_hours < 0 THEN 1 ELSE 0 END) AS invalid_resolution_time,
    SUM(CASE WHEN escalated_flag NOT IN (0, 1) THEN 1 ELSE 0 END) AS invalid_target
FROM customer_case_data;
