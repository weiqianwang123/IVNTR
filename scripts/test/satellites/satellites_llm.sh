export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 0
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env satellites --approach ivntr \
        --seed $seed --offline_data_method "demo" \
        --disable_harmlessness_check True \
        --excluded_predicates "ViewClear,IsCalibrated,HasChemX,HasChemY,Sees" \
        --neupi_pred_config "predicators/config/satellites/pred.yaml" \
        --neupi_gt_ae_matrix False \
        --sesame_task_planner "fdsat" \
        --exclude_domain_feat "none" \
        --neupi_do_normalization False \
        --num_train_tasks 500 \
        --domain_aaai_thresh 300000 \
        --neupi_entropy_w 0.5 \
        --neupi_loss_w 0.5 \
        --neupi_equ_dataset 1.0 \
        --neupi_pred_search_dataset 1.0 \
        --bilevel_plan_without_sim False \
        --sesame_max_samples_per_step 30 \
        --load_approach \
        --load_neupi_from_json True \
        --timeout 5 \
        --approach_dir "saved_approaches/open_models/satellites/ivntr_$seed" \
        --neupi_save_path "saved_approaches/demo/satellites/ivntr_$seed" \
        --log_file logs/satellites/ivntr_ood_test_$seed.log; then
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

    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env satellites --approach ivntr \
        --seed $seed --offline_data_method "demo" \
        --disable_harmlessness_check True \
        --excluded_predicates "ViewClear,IsCalibrated,HasChemX,HasChemY,Sees" \
        --neupi_pred_config "predicators/config/satellites/pred.yaml" \
        --neupi_gt_ae_matrix False \
        --sesame_task_planner "fdsat" \
        --exclude_domain_feat "none" \
        --neupi_do_normalization False \
        --num_train_tasks 500 \
        --neupi_entropy_w 0.5 \
        --neupi_loss_w 0.5 \
        --load_data \
        --neupi_equ_dataset 1.0 \
        --neupi_pred_search_dataset 1.0 \
        --bilevel_plan_without_sim False \
        --execution_monitor expected_atoms \
        --load_approach \
        --load_neupi_from_json False \
        --in_domain_test True \
        --timeout 5 \
        --approach_dir "saved_approaches/demo/satellites/ivntr_llm_$seed" \
        --neupi_load_pretrained "saved_approaches/demo/satellites/ivntr_llm_$seed" \
        --log_file logs/satellites/ivntr_indomain_llm_$seed.log; then
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