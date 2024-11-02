#!/bin/bash

MAIN_LOGFILE="_CONTEXTUALIZE_LOG.log"
CORES_TO_USE=2
MEMORY_KEEPFREE="20G"

contextualize_city() {
    CITY_GPKG_FILE="$1"
    CITY_LOGFILE="${CITY_GPKG_FILE%.gpkg}.log"

    citypy contextualize "$CITY_GPKG_FILE" --regular >> "$CITY_LOGFILE"
}

export -f contextualize_city

parallel -j $CORES_TO_USE --bar --memfree $MEMORY_KEEPFREE --colsep ',' --joblog "$MAIN_LOGFILE" --resume contextualize_city ::: "$(ls ./*.gpkg)"
