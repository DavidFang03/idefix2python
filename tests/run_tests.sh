script_dir="$(CDPATH= cd -- "$(dirname "$0")" && pwd)" 
for f in "$script_dir"/*.py; do python "$f" -j 8; done