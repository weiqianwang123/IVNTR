export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Optional: Set to "realworld" to use real-world data, or "oracle" for simulated data
DATA_SOURCE="realworld"  # Change to "realworld" to use real-world demonstrations

# Number of real-world demonstrations to use (if DATA_SOURCE="realworld")
REALWORLD_NUM_DEMOS=10 # Adjust based on how many demos you've converted

for seed in 0
do
    echo "Running Seed $seed --------------------------------------"
    
    # Configure data source
    if [ "$DATA_SOURCE" = "realworld" ]; then
        echo "Using real-world demonstration data with $REALWORLD_NUM_DEMOS demonstrations"
        DATA_SUFFIX="realworld"
        NUM_DEMOS=$REALWORLD_NUM_DEMOS
        
        # First, ensure the real-world data is in the expected location
        # The system expects: saved_datasets/clean-table-real__demo__oracle__NUM____SEED__None.data
        EXPECTED_FILE="saved_datasets/clean-table-real-real__demo__oracle__${NUM_DEMOS}____${seed}__None.data"
        REALWORLD_FILE="saved_datasets/clean-table-real-real__demo__realworld__${NUM_DEMOS}____0__None.data"
        
        if [ -f "$REALWORLD_FILE" ]; then
            echo "Copying real-world data to expected location: $EXPECTED_FILE"
            cp "$REALWORLD_FILE" "$EXPECTED_FILE"
        else
            echo "ERROR: Real-world data file not found: $REALWORLD_FILE"
            echo "Please run: python convert_realworld_to_demo.py --num-demos $NUM_DEMOS"
            exit 1
        fi
    else
        echo "Using oracle demonstration data"
        DATA_SUFFIX="oracle"
        NUM_DEMOS=60
    fi
    
    # Record start time
    start_time=$(date +%s)
    # Training clean table real environment with PDLM approach
    if python3 predicators/main.py --env clean-table-real-real --approach ivntr-pdlm \
        --seed $seed --offline_data_method "demo" \
        --disable_harmlessness_check True \
        --excluded_predicates "toy_on_table,handempty,holdingToy,toy_in_box,wiper_in_box,wiper_on_table,holdingWiper,box_at_center,box_at_side,No_toy_at_table,table_clean" \
        --neupi_pred_config "predicators/config/clean_table_real/pred_pdlm.yaml" \
        --pred_pddl_config "predicators/config/clean_table_real/pddl.json" \
        --neupi_gt_ae_matrix False \
        --sesame_task_planner "fdsat" \
        --exclude_domain_feat "none" \
        --neupi_do_normalization False \
        --num_train_tasks $NUM_DEMOS \
        --load_data \
        --domain_aaai_thresh 150000 \
        --neupi_entropy_w 0.5 \
        --neupi_loss_w 0.5 \
        --neupi_equ_dataset 1.0 \
        --neupi_pred_search_dataset 1.0 \
        --bilevel_plan_without_sim False \
        --sesame_max_samples_per_step 20 \
        --timeout 30 \
        --approach_dir "saved_approaches/demo/clean-table-real/ivntr_pdlm_${DATA_SUFFIX}_$seed" \
        --neupi_save_path "saved_approaches/demo/clean-table-real/ivntr_pdlm_${DATA_SUFFIX}_$seed" \
        --log_file logs/clean-table-real/ivntr_train_pdlm_${DATA_SUFFIX}_$seed.log; then
        echo "Seed $seed completed successfully."
    else
        echo "Seed $seed encountered an error."
    fi

    # Record end time
    end_time=$(date +%s)

    # Calculate the duration in seconds
    runtime=$((end_time - start_time))

    # Convert to hours, minutes, and seconds
    hours=$((runtime / 3600))
    minutes=$(( (runtime % 3600) / 60 ))
    seconds=$((runtime % 60))

    # Output the total runtime
    echo "Seed $seed completed in: ${hours}h ${minutes}m ${seconds}s"
done