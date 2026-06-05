
import geotessera

# Generate a coverage map showing all available tiles
geotessera.coverage --year 2024 --output tessera_global_2024.png

# Generate a coverage map for the UK
geotessera coverage --country uk

# View coverage for a specific year
geotessera coverage --year 2024 --output coverage_2024.png