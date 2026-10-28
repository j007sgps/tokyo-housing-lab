-- 學區（load 時從 XKT004、XKT005 建）
SELECT kind, code, name, city, owner, address, polygons, outlines
FROM school_district
ORDER BY kind, city, code;
