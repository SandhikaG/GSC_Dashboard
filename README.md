# GSC Data Fetcher

This project fetches data from Google Search Console (GSC) and stores it in a database, offering options for both PostgreSQL and Supabase REST API storage. It's designed to be run as a scheduled task, automatically collecting recent data and providing historical data fetching capabilities. The project uses a service account for authentication with the Google Search Console API.

## Features

- **Data Fetching**: Retrieves search analytics data (clicks, impressions, CTR, position) from Google Search Console.
- **Storage Options**: Supports storing data in a PostgreSQL database or via Supabase REST API.
- **Historical Data**: Can fetch historical data for a specified number of months.
- **Daily Updates**: Designed for daily updates to capture recent data.
- **Connection Management**: Includes retry logic and IPv4 preference for database connections.
- **Error Handling**: Implements error handling for GSC API and database operations.
- **Database Management**: Creates the necessary database table if it doesn't exist.
- **REST API Support**: Option to use Supabase REST API for data storage, useful when direct database connection is problematic.
- **Github Actions**: Automates data fetching and storage.

## Usage

### Prerequisites

- Python 3.11 or higher
- Required Python packages (see [requirements.txt](https://github.com/naveen3830/gsc-api-data/blob/main/requirements.txt) or [pyproject.toml](https://github.com/naveen3830/gsc-api-data/blob/main/pyproject.toml))
- Google Cloud Service Account key in JSON format
- Google Search Console access for the service account

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd gsc-api-data
    ```
2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  Set the following environment variables:
    *   `SERVICE_ACCOUNT_FILE`: Path to the Google Cloud Service Account JSON file (e.g., `./service_account.json`).
    *   `SITE_URL`: The URL of the website in Google Search Console (e.g., `https://example.com`).
    *   `DATABASE_URL`: The connection string for the PostgreSQL database (e.g., `postgresql://user:password@host:port/database`).
    *   `SUPABASE_URL`: The URL of your Supabase project (e.g., `https://your-project.supabase.co`).
    *   `SUPABASE_ANON_KEY`: The anonymous key for your Supabase project.
    *   `USE_REST_API`: Set to `"true"` to use the Supabase REST API instead of PostgreSQL (optional, defaults to `"false"`).
2.  For local development, you can create a `.env` file in the project root and add the variables there.  The `.env` file is automatically loaded by the application, refer to the [main.py](https://github.com/naveen3830/gsc-api-data/blob/main/main.py) file.

### Running the script

The `main.py` script accepts the following arguments:

*   `--mode`: Operation mode (`historical`, `daily`, `test`, `summary`, `test-db`). Defaults to `daily`.
*   `--months`: Number of months for historical data fetch (only applicable in `historical` mode). Defaults to 16.
*   `--days`: Number of days for daily data fetch (only applicable in `daily` mode). Defaults to 7.

Examples:

*   Run in daily mode:

    ```bash
    python main.py --mode daily
    ```
*   Run in historical mode, fetching data for the last 6 months:

    ```bash
    python main.py --mode historical --months 6
    ```
*   Run in test mode to test GSC and database connections:

    ```bash
    python main.py --mode test
    ```

### GitHub Actions

The project includes a [GitHub Actions workflow](https://github.com/naveen3830/gsc-api-data/blob/main/.github/workflows/gsc.yml) for automated data fetching. To set up the workflow:

1.  Add the following secrets to your GitHub repository:
    *   `GSC_SERVICE_ACCOUNT_JSON`: The contents of your Google Cloud Service Account JSON file.
    *   `SITE_URL`: The URL of the website in Google Search Console.
    *   `DATABASE_URL`: The connection string for the PostgreSQL database or leave empty if using REST API.
    *   `SUPABASE_URL`: The URL of your Supabase project.
    *   `SUPABASE_ANON_KEY`: The anonymous key for your Supabase project.

2.  The workflow is scheduled to run daily at 6 AM UTC. You can adjust the schedule in the `.github/workflows/gsc.yml` file.  You can also manually trigger the workflow from the GitHub Actions tab.

## Database Schema

The data is stored in a table named `gsc_metrics` with the following schema:

| Column      | Data Type      | Description                                 |
| ----------- | -------------- | ------------------------------------------- |
| date        | DATE           | Primary key, representing the date of data |
| clicks      | INTEGER        | Number of clicks                            |
| impressions | INTEGER        | Number of impressions                       |
| ctr         | DECIMAL(10, 6) | Click-through rate                          |
| position    | DECIMAL(10, 2) | Average position                            |
| created\_at | TIMESTAMP      | Timestamp of when the record was created    |
| updated\_at | TIMESTAMP      | Timestamp of when the record was last updated |

## REST API

When `USE_REST_API` is set to `"true"`, the project uses the Supabase REST API to interact with the database. Ensure that the `SUPABASE_URL` and `SUPABASE_ANON_KEY` environment variables are properly configured.  The `test_rest_api_connection` function in [main.py](https://github.com/naveen3830/gsc-api-data/blob/main/main.py) can be used to check REST API connectivity.

## Error Handling

The script includes error handling for various scenarios, such as:

*   GSC API connection errors
*   Database connection errors
*   HTTP errors from the GSC API
*   Database insertion errors

## clear\_database.py

The [clear_database.py](https://github.com/naveen3830/gsc-api-data/blob/main/clear_database.py) script can be run to truncate the table `gsc_metrics` and reset the identity.
