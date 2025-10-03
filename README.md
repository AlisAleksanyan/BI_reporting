# Agenda Management Reporting

A Streamlit web application for managing and analyzing retail agenda data.
This demo uses fake sample data, but the original project was built with real company data.
It includes the full data processing pipeline: from raw extraction, cleaning, and transformation to creating readable reports and dashboards.

### Features

Automated ETL pipeline: raw data → cleaned & structured datasets (via GitHub Actions)

Live data refresh from GitHub (via GitHub API)

Interactive tables & charts for shop and area manager performance

Daily automated reporting with export options

User-friendly filters: date, region, shop, area manager

### Real Project (Production Setup)

The real project was significantly more advanced and enabled near real-time reporting:

1. Source: ERP via SQL

Extracted raw agenda and operations data directly from the company ERP.

Multiple relational tables used (appointments, shops, resources, area mappings).

2. Automation: Power Automate

Scheduled flows automatically refreshed SQL extracts throughout the day.

Guaranteed the pipeline stayed in sync with ERP system changes.

3. Integration: Cron Jobs + GitHub

A cron scheduler pushed processed datasets into GitHub.

GitHub acted as the single “source of truth” for the reporting app.

4. Reporting Layer: Streamlit App

Streamlit app fetched data from GitHub API in near real time.

Provided dashboards, KPIs, and exports directly to business users.

### Result: Business users gained access to real-time agenda reporting, replacing manual spreadsheets with a fully automated, auditable, and always up-to-date solution.
