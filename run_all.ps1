Write-Host "Step 1: Fetching data from Kraken..."
py fetch_data.py

Write-Host "Step 2: Training the LSTM model..."
py train_model.py

Write-Host "Step 3: Generating evaluation report..."
py update_eval_table.py

Write-Host "Finished! Check out EVAL.md and /images :)"