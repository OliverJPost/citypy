#!/bin/bash

MAIN_LOGFILE="_PROCESS_LOG"
CONTEXT_LOGFILE="_CONTEXT_LOG"
CITY_FOLDER="."
MEMORY_KEEPFREE="10G"

SMALL_CORES=12 # Supposedly under 20GB, n=31 12x
SMALL_CORES_CONTEXT=8
MEDIUM_CORES=8 # n=3+5=8, supposedly under 35GB -> 6x
MEDIUM_CORES_CONTEXT=4
LARGE_CORES=6 # Supposedly under 50GB, n =6 -> 4x
LARGE_CORES_CONTEXT=3
EXTREME_CORES=3 # Supposedly under 120GB, n =3 -> 2x
EXTREME_CORES_CONTEXT=1

#SMALL_SIZE=1000000   # 1GB in KB
#MEDIUM_SIZE=2000000  # 2GB in KB
#EXTREME_SIZE=2800000 # 2.8GB in KB
SMALL_SIZE=3000000   # 1GB in KB
MEDIUM_SIZE=6000000  # 2GB in KB
EXTREME_SIZE=1000000 # 2.8GB in KB
process_city() {
    CITY_GPKG_FILE="$1"
    CITY_LOGFILE="${CITY_GPKG_FILE%.gpkg}.log"

    citypy process "$CITY_GPKG_FILE" >> "$CITY_LOGFILE" 2>&1 -s building_class
}

export -f process_city

contextualize_city() {
    CITY_GPKG_FILE="$1"
    CITY_LOGFILE="${CITY_GPKG_FILE%.gpkg}.log"

    citypy contextualize "$CITY_GPKG_FILE" --buildings >> "$CITY_LOGFILE" 2>&1 && citypy contextualize "$CITY_GPKG_FILE" --regular >> "$CITY_LOGFILE" 2>&1 && citypy contextualize "$CITY_GPKG_FILE" --major >> "$CITY_LOGFILE" 2>&1
}
export -f contextualize_city

# Separate files based on size
find "$CITY_FOLDER" -type f -name "*.gpkg" -print0 | while IFS= read -r -d '' file; do
    FILE_SIZE=$(du -k "$file" | cut -f1)
    if [ "$FILE_SIZE" -lt "$SMALL_SIZE" ]; then
        echo "$FILE_SIZE $file" >> small_files_unsorted.txt
    elif [ "$FILE_SIZE" -lt "$MEDIUM_SIZE" ]; then
        echo "$FILE_SIZE $file" >> medium_files_unsorted.txt
    elif [ "$FILE_SIZE" -lt "$EXTREME_SIZE" ]; then
        echo "$FILE_SIZE $file" >> large_files_unsorted.txt
    else
        echo "$FILE_SIZE $file" >> extreme_files_unsorted.txt
    fi
done

# Sort the files by size in descending order and store in final lists
sort -nr small_files_unsorted.txt | awk '{print $2}' > small_files.txt
sort -nr medium_files_unsorted.txt | awk '{print $2}' > medium_files.txt
sort -nr large_files_unsorted.txt | awk '{print $2}' > large_files.txt
sort -nr extreme_files_unsorted.txt | awk '{print $2}' > extreme_files.txt

# Remove unsorted files
rm small_files_unsorted.txt medium_files_unsorted.txt large_files_unsorted.txt extreme_files_unsorted.txt

# Process small files
if [ -f small_files.txt ]; then
    parallel -j $SMALL_CORES --bar --progress --memfree $MEMORY_KEEPFREE --joblog "${MAIN_LOGFILE}_small.log" process_city :::: small_files.txt
    #parallel -j $SMALL_CORES_CONTEXT --bar --progress --memfree $MEMORY_KEEPFREE --joblog "${CONTEXT_LOGFILE}_small.log" contextualize_city :::: small_files.txt
    rm small_files.txt
fi

# Process medium files
if [ -f medium_files.txt ]; then
    parallel -j $MEDIUM_CORES --bar --progress --memfree $MEMORY_KEEPFREE --joblog "${MAIN_LOGFILE}_medium.log" process_city :::: medium_files.txt
    #parallel -j $MEDIUM_CORES_CONTEXT --bar --progress --memfree $MEMORY_KEEPFREE --joblog "${CONTEXT_LOGFILE}_medium.log" contextualize_city :::: medium_files.txt
    rm medium_files.txt
fi

# Process large files
if [ -f large_files.txt ]; then
    parallel -j $LARGE_CORES --bar --progress --memfree $MEMORY_KEEPFREE --joblog "${MAIN_LOGFILE}_large.log" process_city :::: large_files.txt
    #parallel -j $LARGE_CORES_CONTEXT --bar --progress --memfree $MEMORY_KEEPFREE --joblog "${CONTEXT_LOGFILE}_large.log" contextualize_city :::: large_files.txt
    rm large_files.txt
fi

# Process extreme files
if [ -f extreme_files.txt ]; then
    parallel -j $EXTREME_CORES --bar --progress --memfree $MEMORY_KEEPFREE --joblog "${MAIN_LOGFILE}_extreme.log" process_city :::: extreme_files.txt
    #parallel -j $EXTREME_CORES_CONTEXT --bar --progress --memfree $MEMORY_KEEPFREE --joblog "${CONTEXT_LOGFILE}_extreme.log" contextualize_city :::: extreme_files.txt
    rm extreme_files.txt
fi



