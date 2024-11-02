#!/bin/bash

MAIN_LOGFILE="_DOWNLOAD_LOG.log"
CORES_TO_USE=3
MEMORY_KEEPFREE="10G"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if the city file path is passed as an argument
if [ "$#" -eq 1 ]; then
    CITY_FILE="$1"
else
    echo "Usage: $0 <city_file_path>"
    exit 1
fi

# Overwrite config to use local Overpass API instance
CITYPY_CUSTOM_API_INSTANCE="{enable=true, endpoint='http://localhost:1414/cgi-bin'}"
export CITYPY_CUSTOM_API_INSTANCE

# Check if the city list file exists
if [ ! -f "$CITY_FILE" ]; then
    echo "Error: City list file '$CITY_FILE' not found."
    exit 1
fi

download_city() {
    CITY="$1"
    COUNTRY="$2"
    WEST="$3"
    SOUTH="$4"
    EAST="$5"
    NORTH="$6"
    BBOX="$WEST,$SOUTH,$EAST,$NORTH"
    CITY_WITHOUT_SPACES="${CITY// /_}"
    COUNTY_WITHOUT_SPACES="${COUNTRY// /_}"
    CITY_LOGFILE="${CITY_WITHOUT_SPACES^}_${COUNTY_WITHOUT_SPACES^^}.log"

    citypy download "$CITY" "$COUNTRY" --bbox "$BBOX" >> "$CITY_LOGFILE" 2>&1
}

export -f download_city

parallel -j $CORES_TO_USE --bar --progress --memfree $MEMORY_KEEPFREE --colsep ',' --joblog "$MAIN_LOGFILE" download_city ::: "$(grep -v '^#' "$CITY_FILE")"