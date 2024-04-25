#!/bin/bash

model='model'
seed=42
epochs=(100 150 200)
lr=1e-2
weight_decay=5e-4
hidden=16
dropout=(0.2 0.3 0.5)

for epoch in "${epochs[@]}"; do
    for drop in "${dropout[@]}"; do
        python -u -m main --model "$model" \
                        --seed "$seed" \
                        --epochs "$epoch" \
                        --lr "$lr" \
                        --weight_decay "$weight_decay" \
                        --hidden "$hidden" \
                        --dropout "$dropout"
    done
done

wait