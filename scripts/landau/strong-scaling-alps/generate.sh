#!/bin/bash

# this is the directory of the script, regardless of where it's called from
if [[ $0 != $BASH_SOURCE ]]; then
  # this script was sourced from somewhere, expand name if a symlink
  SCRIPT_DIR=$(dirname $(readlink -f $BASH_SOURCE))
else
  # this was executed directly
  SCRIPT_DIR="$(readlink -f $(dirname $0))"
fi

# usual node counts
nodes=(004 008 016 032 064 128 256)
current_dir=$PWD
script_name=@LANDAU_SCRIPT_NAME@

for g in "${nodes[@]}"; do
  jobdir="strongscaling_landau/nodes_$g"
  echo "Generating job for node count $g in $current_dir/$jobdir"
  mkdir -p "$current_dir/$jobdir"
  cp "$SCRIPT_DIR/$script_name" "$jobdir/$script_name"
  sed -i "s/_n_/$g/g" "$current_dir/$jobdir/$script_name"
  cd "$jobdir"
  sbatch $script_name
  cd "$current_dir"
done

