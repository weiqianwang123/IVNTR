#!/bin/bash
export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Test online planning with a trained IVNTR model
python3 predicators/main.py \
  --env clean-table-real-real \
  --approach ivntr \
  --seed 0 \
  --online_planning \
  --state_input example_state.json \
  --load_approach \
  --approach_dir "saved_approaches/demo/clean-table-real/ivntr_realworld_0" \
  --neupi_save_path "saved_approaches/demo/clean-table-real/ivntr_realworld_0" \
  --neupi_pred_config "predicators/config/clean_table_real/pred_pdlm.yaml" \
  --pred_pddl_config "predicators/config/clean_table_real/pddl.json" \
  --timeout 30 \
  --excluded_predicates "toy_on_table,handempty,holdingToy,toy_in_box,wiper_in_box,wiper_on_table,holdingWiper,box_at_center,box_at_side,No_toy_at_table,table_clean"