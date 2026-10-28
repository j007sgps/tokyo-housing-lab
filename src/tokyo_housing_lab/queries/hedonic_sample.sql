-- 品質調整指數（hedonic 迴歸）的樣本：中古マンション等，一列一筆成交
--   排除：㎡ 單價無法計算、建築年不明或晚於成交年 1 年以上、備註有特殊事情（調停・競売等、関係者間取引…）
--   （預售或剛完工的物件，建築年可能比成交年晚 1 年，屋齡當成 0）
--   $region：'13' = 東京都全域
SELECT
    city,
    coalesce(district_name, '') AS district,
    year,
    quarter,
    price_classification,
    ln(unit_price_per_m2) AS log_price,
    area_m2 AS area,
    building_year,
    coalesce(floor_plan, '') AS floor_plan,
    coalesce(structure, '') AS structure,
    coalesce(renovation, '') AS renovation,
    coalesce(city_planning, '') AS city_planning,
    CASE WHEN floor_area_ratio_status = 'ok' THEN floor_area_ratio END AS floor_area_ratio
FROM transactions
WHERE starts_with(city, $region)
  AND property_type = '中古マンション等'
  AND unit_price_per_m2_status = 'ok'
  AND area_m2_status = 'ok'
  AND building_year_status = 'ok'
  AND building_year <= year + 1
  AND coalesce(remarks, '') = ''
ORDER BY city, year, quarter, price_classification, raw_file, row_index;
