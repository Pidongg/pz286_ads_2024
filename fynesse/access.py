from .config import *
import requests
import pymysql
import csv
import osmnx as ox
from pyrosm import OSM
import pandas as pd
import os
import csv
from typing import Dict, List, Optional
from IPython import get_ipython
import geopandas as gpd
import zipfile
import io

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """


def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print(f"Downloading data for year: {year}")
        for part in range(1, 3):
            url = base_url + \
                file_name.replace("<year>", str(year)).replace(
                    "<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)


def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn


def housing_upload_join_data(conn, year):
    start_date = str(year) + "-01-01"
    end_date = str(year) + "-12-31"

    cur = conn.cursor()
    print('Selecting data for year: ' + str(year))
    cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' +
                start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
    rows = cur.fetchall()

    csv_file_path = 'output_file.csv'

    # Write the rows to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the data rows
        csv_writer.writerows(rows)
    print('Storing data for year: ' + str(year))
    cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path +
                "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
    conn.commit()
    print('Data stored for year: ' + str(year))


def _download_census_data(code, year='2021', base_dir='', level='oa'):
    url = f'https://www.nomisweb.co.uk/output/census/{year}/census{year}-{code.lower()}.zip'
    if year == '2011':
        url = f'https://www.nomisweb.co.uk/output/census/{
            year}/{code.lower()}_2011_{level}.zip'
    extract_dir = os.path.join(
        base_dir, os.path.splitext(os.path.basename(url))[0])

    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"Files already exist at: {extract_dir}.")
        return

    os.makedirs(extract_dir, exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Files extracted to: {extract_dir}")


def _load_census_data(code, year='2021', level='oa'):
    if year == '2011':
        if code[0] == 'K':
            return pd.read_csv(f'{code.lower()}_{year}_{level}/{code.upper()}DATA.csv')
        return pd.read_csv(f'{code.lower()}_{year}_{level}/{code.lower()}_{year}_{level}/{code.upper()}DATA.csv')
    return pd.read_csv(f'census{year}-{code.lower()}/census{year}-{code.lower()}-{level}.csv')


def retrieve_census_data(code, year='2021', level='oa'):
    _download_census_data(code, year)
    return _load_census_data(code, year, level)


def get_osm_data(center_latitude, center_longitude, box_size_km, tags):
    # TODO: fix having a fixed denominator of 111
    north = center_latitude + box_size_km/(2*111)
    south = center_latitude - box_size_km/(2*111)
    west = center_longitude - box_size_km/(2*111)
    east = center_longitude + box_size_km/(2*111)
    new_pois = ox.geometries_from_bbox(north, south, east, west, tags)
    return new_pois


class OSMDataManager:
    def __init__(self, output_dir: str = "./"):
        self.output_dir = output_dir

    def download_osm_data(self, region: str = "great-britain") -> str:
        """
        Download OpenStreetMap data for a specified region
        """
        url = f"https://download.geofabrik.de/europe/{region}-latest.osm.pbf"
        output_file = os.path.join(self.output_dir, f'{region}-latest.osm.pbf')

        if os.path.exists(output_file):
            print(f"File already exists at: {output_file}")
            return output_file

        print(f"Downloading {region} OSM data...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("Download complete!")
        return output_file

    def get_pois(self,
                 pbf_file: str,
                 custom_filter: Dict[str, List[str]],
                 columns: Optional[List[str]] = None) -> gpd.GeoDataFrame:
        """
        Extract POIs from OSM file with custom filters
        """
        try:
            osm = OSM(pbf_file)
            pois = osm.get_pois(custom_filter=custom_filter)

            if columns and isinstance(pois, gpd.GeoDataFrame):
                available_cols = [
                    col for col in columns if col in pois.columns]
                pois = pois[available_cols]

            return pois

        except Exception as e:
            print(f"Error reading OSM data: {e}")
            return None

    def column_filter(self, pois: gpd.GeoDataFrame, columns_to_keep: List[str]) -> gpd.GeoDataFrame:
        return pois[columns_to_keep]


def load_chunks_to_table(df: pd.DataFrame,
                         table_name: str,
                         columns: List[str],
                         chunk_size: int = 1000,
                         geometry_column: Optional[str] = None,
                         conn=None) -> None:
    """
    Load dataframe into MySQL table in chunks, with special handling for geometry data

    Args:
        df (pd.DataFrame): DataFrame to load
        table_name (str): Name of the target table
        columns (List[str]): List of column names to load (excluding geometry)
        chunk_size (int): Size of chunks to process
        geometry_column (Optional[str]): Name of the geometry column, if any
        conn: Database connection object
    """
    df = df.reset_index(drop=True)
    def load_chunk(chunk: pd.DataFrame, chunk_num: int) -> None:
        csv_file = f'{table_name}_chunk_{chunk_num}.csv'
        try:
            if geometry_column:
                # Handle geometry column if present
                chunk = chunk.dropna(subset=[geometry_column])
                chunk[geometry_column] = chunk[geometry_column].astype(
                    str).str.replace(',', 'and')

            # Save to CSV with explicit handling
            chunk.to_csv(csv_file,
                         index=False,
                         na_rep='\\N',
                         quoting=csv.QUOTE_MINIMAL
                         )

            # Build the column list for SQL
            if geometry_column:
                col_list = columns + [f'@{geometry_column}']
                geometry_set = f"SET {
                    geometry_column} = TRIM(BOTH '\"' FROM @{geometry_column})"
            else:
                col_list = columns
                geometry_set = ""

            # Load data with SQL
            load_sql = f"""
            LOAD DATA LOCAL INFILE '{csv_file}'
            INTO TABLE {table_name}
            FIELDS TERMINATED BY ','
            ENCLOSED BY '"'
            LINES TERMINATED BY '\n'
            IGNORE 1 LINES
            ({', '.join(col_list)})
            {geometry_set};
            """

            if conn:
                with conn.cursor() as cursor:
                    cursor.execute(load_sql)
                conn.commit()
            else:
                ipython = get_ipython()
                ipython.run_line_magic('sql', load_sql)

        except Exception as e:
            print(f"Error processing chunk {chunk_num}:", e)
            if geometry_column:
                print("\nSample of problematic data:")
                print(chunk[geometry_column].head())
        finally:
            if os.path.exists(csv_file):
                os.remove(csv_file)

    # Process in chunks
    for i in range(0, len(df), chunk_size):
        load_chunk(df.iloc[i:i+chunk_size], i)


def setup_spatial_columns(conn, table_name: str,
                          text_geometry_col: str = 'geometry',
                          spatial_geometry_col: str = 'geometry_col') -> None:
    """
    Set up spatial columns and indexes for a table with geometry data

    Args:
        conn: Database connection object
        table_name (str): Name of the table to modify
        text_geometry_col (str): Name of the column containing geometry text
        spatial_geometry_col (str): Name of the spatial column to create
    """
    try:
        with conn.cursor() as cursor:
            # Clean up geometry text
            cursor.execute(f"""
                UPDATE {table_name}
                SET {text_geometry_col} = REPLACE(
                    TRIM(BOTH '"' FROM {text_geometry_col}),
                    'and',
                    ','
                );
            """)

            # Add spatial column if it doesn't exist
            cursor.execute(f"""
                ALTER TABLE {table_name}
                ADD COLUMN IF NOT EXISTS {spatial_geometry_col} GEOMETRY NOT NULL;
            """)

            # Convert text to geometry
            cursor.execute(f"""
                UPDATE {table_name}
                SET {spatial_geometry_col} = ST_GeomFromText({text_geometry_col})
                WHERE {text_geometry_col} IS NOT NULL;
            """)

            # Create spatial index if it doesn't exist
            cursor.execute(f"""
                CREATE SPATIAL INDEX IF NOT EXISTS idx_{spatial_geometry_col}
                ON {table_name}({spatial_geometry_col});
            """)

        conn.commit()
        print(f"Successfully set up spatial columns for {table_name}")

    except Exception as e:
        print(f"Error setting up spatial columns: {e}")
        conn.rollback()


def setup_sql_magic(username: str, password: str, url: str, database: str = 'ads_2024') -> None:
    """
    Set up SQL magic configuration and connection for Jupyter notebooks

    Args:
        username (str): Database username
        password (str): Database password
        url (str): Database URL
        database (str): Database name to use (default: 'ads_2024')
    """
    try:
        ipython = get_ipython()

        # Configure SQL magic
        ipython.run_line_magic(
            'config', "SqlMagic.style = '_DEPRECATED_DEFAULT'")

        # Set up connection string
        connection_string = f"mariadb+pymysql://{
            username}:{password}@{url}?local_infile=1"
        ipython.run_line_magic('sql', connection_string)

        # Use specified database
        ipython.run_line_magic('sql', f'USE `{database}`')

        # Set SQL mode and timezone
        ipython.run_line_magic('sql', 'SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO"')
        ipython.run_line_magic('sql', 'SET time_zone = "+00:00"')

        # Show available tables
        result = ipython.run_line_magic('sql', 'SHOW TABLES')
        print("\nAvailable tables:")
        for row in result:
            print(f"- {row[0]}")

    except Exception as e:
        print(f"Error setting up SQL magic: {e}")


def load_table(table_name: str, conn):
    """
    Load and join all necessary data tables to associate houses with OAs and census data
    """
    final_query = f"""
        SELECT * from {table_name}
    """

    # Execute query
    df = pd.read_sql(final_query, conn)

    return df
