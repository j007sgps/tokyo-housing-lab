# tokyo-housing-lab

東京中古公寓行情分析工具（個人學習用）。資料來源為國土交通省的公開 API（XIT001）。

目前進度：M1（立川市 2021Q1〜最新季的成交資料 → DuckDB）完成。

## 環境準備

```bash
cd tokyo-housing-lab
uv sync
cp .env.example .env   # 再把 REINFOLIB_API_KEY 換成自己的金鑰
```

金鑰也可以直接用環境變數 `REINFOLIB_API_KEY` 提供（環境變數優先於 `.env`）。

## 使用方式

```bash
uv run tokyo-housing-lab fetch   # 打 API，原始 JSON 存到 data/raw/（已有的跳過，有資料的最新 4 季重抓）
uv run tokyo-housing-lab load    # 只讀 data/raw/，清理後寫進 data/tokyo_housing_lab.duckdb（不打 API）
uv run tokyo-housing-lab query   # 每季 ㎡ 單價中位數與筆數（中古マンション等）
```

- `fetch --force`：全部重抓；`fetch --refresh-recent 0`：不重抓最新幾季；`fetch --start 2023Q1`：改起始季
- `load` 每次都把 `data/raw/` 整個重新載入，跑幾次結果都一樣；任何一個檔案有問題就整批不寫入

查詢用的 SQL 在 `src/tokyo_housing_lab/queries/quarterly_median.sql`，也可以直接用 DuckDB 開資料庫自己查：

```bash
uv run python -c "import duckdb; print(duckdb.connect('data/tokyo_housing_lab.duckdb', read_only=True).sql('SELECT count(*) FROM transactions'))"
```

資料表：
- `transactions`：中古マンション等的成交資料。數值欄位都有 `<欄位>_raw`（原字串）、`<欄位>`（數值）、`<欄位>_status`（ok / empty / sentinel / unparsable）
- `fetch_log`：每次打 API 的紀錄（含 404 的「該期無資料」）

欄位的實際觀察與清理規則的依據：`docs/xit001_fields.md`

## 測試

```bash
uv run pytest            # 單元測試 + 用 fixture 的整條管線測試（不會打 API）
uv run pytest -m manual  # 真的打 API 的整合測試（需要金鑰與網路）
```

## 資料目錄

- `data/raw/`：API 回傳的原始 JSON（不進 git）
- `data/*.duckdb`：清理後的資料庫（不進 git，可由 raw 重建）

---

このサービスは、国土交通省の不動産情報ライブラリのAPI機能を使用していますが、提供情報の最新性、正確性、完全性等が保証されたものではありません
