export API_KEY="your_api_key"
python3 src/inference.py \
    --model_path "" \
    --lora_path "" \
    --use_flash_attention \
    --acceleration "" \
    --model_class "GPT41" \
    --task_list "train_set/MedMCQA" \
    --batch_size 8 \
    --data_path DGRC_benchmark/dataset \
    --save_path results_teacher \
    --save_id gpt41 \
    --samples 1