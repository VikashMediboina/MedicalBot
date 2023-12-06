autotrain llm \
    --train \
    --model TinyPixel/Llama-2-7B-bf16-sharded \
    --project-name medical-bot \
    --data-path /home/mediboina.v/Vikash/medicalBot/Data \
    --text-column Text \
    --lr 2e-4 \
    --batch-size 1 \
    --epochs 3 \
    --trainer sft \
    --push_to_hub \
    --token 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 \
    --repo_id vikashmediboina/medical-bot \
    --block_size 32 \
    --disable_gradient_checkpointing \
    --max_grad_norm 0.5 \
    --logging_steps 20 \
    --warmup_ratio 0.1 \
    --optimizer adamw_torch \
    --scheduler linear \
    --weight_decay 0.01 \
    --use_int8