export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 0
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env tools-pcd --approach oracle \
        --seed $seed --offline_data_method "demo" \
        --disable_harmlessness_check True \
        --num_train_tasks 500 \
        --timeout 5 \
        --approach_dir "saved_approaches/open_models/tools-pcd/ivntr_$seed" \
        --neupi_save_path "saved_approaches/open_models/tools-pcd/ivntr_$seed" \
        --log_file logs/tools-pcd/ivntr_ood_test_oracle_$seed.log; then
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