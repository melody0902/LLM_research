#!/bin/bash
set -e

OUTPUT_DIR="outputs/0123_200green"
TOPK=30

for METRIC in raw norm
do
  SAVE_DIR="line_all_${METRIC}_clean"
  echo "▶ Generating all plots: metric=${METRIC}"
  python3 visualize_wm_token_line.py \
    --output_dir ${OUTPUT_DIR} \
    --save_dir ${SAVE_DIR} \
    --topk ${TOPK} \
    --mode both \
    --metric ${METRIC}
done

echo " All scenarios generated"


# chmod +x run_all_token_bar_plots.sh
# ./run_all_token_bar_plots.sh

