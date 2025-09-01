export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 0
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # Training clean table real environment with PDLM approach
    if python3 predicators/main.py --env clean-table-real --approach oracle \
        --seed $seed --offline_data_method "demo" \
        --disable_harmlessness_check True \
        --neupi_pred_config "predicators/config/clean_table_real/pred_pdlm.yaml" \
        --pred_pddl_config "predicators/config/clean_table_real/pddl.json" \
        --neupi_gt_ae_matrix False \
        --sesame_task_planner "fdsat" \
        --exclude_domain_feat "none" \
        --neupi_do_normalization False \
        --num_train_tasks 60 \
        --domain_aaai_thresh 150000 \
        --neupi_entropy_w 0.5 \
        --neupi_loss_w 0.5 \
        --neupi_equ_dataset 1.0 \
        --neupi_pred_search_dataset 1.0 \
        --bilevel_plan_without_sim False \
        --sesame_max_samples_per_step 20 \
        --timeout 30 \
        --approach_dir "saved_approaches/demo/clean-table-real/ivntr_pdlm_sim_$seed" \
        --neupi_save_path "saved_approaches/demo/clean-table-real/ivntr_pdlm_sim_$seed" \
        --log_file logs/clean-table-real/ivntr_train_pdlm_sim_$seed.log; then
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