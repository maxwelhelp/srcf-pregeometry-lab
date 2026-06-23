# Команды запуска

## 1. Controlled hard compose test

```bash
python experiments/circuit_signal_birth_qwen/signal_birth_test_v1_3_hard.py \
  --device cuda \
  --runs 5 \
  --n-train 4096 \
  --n-heldout 2048 \
  --n-retain 2048 \
  --hidden-signals 2 \
  --max-births 2 \
  --max-depth 4 \
  --spurious-count 16 \
  --print-top 12
```

## 2. Real Qwen circuit signal birth

```bash
python experiments/circuit_signal_birth_qwen/real_qwen_circuit_signal_birth_v1_2.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --eval-device cuda \
  --dtype fp16 \
  --attn-implementation eager \
  --layers 2,6,23 \
  --max-length 64 \
  --max-delta 63 \
  --prompts-per-suite 1 \
  --target-split heads \
  --qk-candidates 6 \
  --qk-births 3 \
  --vo-mode per-head-prompts \
  --vo-per-head-candidates 2 \
  --vo-per-head-births 1 \
  --vo-include-full-residual \
  --vo-min-held-y-gain 0.001
```

## 3. Joint all-head circuit benchmark

```bash
python experiments/circuit_signal_birth_qwen/real_qwen_joint_heads_circuit_compare_v2.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --eval-device cuda \
  --dtype fp16 \
  --attn-implementation eager \
  --layers 2,6,23 \
  --max-length 64 \
  --max-delta 63 \
  --prompts-per-suite 1 \
  --rank-raw 16 \
  --rank-circuit-qk 1 \
  --rank-circuit-vo 16 \
  --print-heads 5
```

## 4. Fine-tune protection stress test

```bash
python experiments/circuit_signal_birth_qwen/circuit_birth_finetune_protect_v1_3.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --dtype fp16 \
  --attn-implementation eager \
  --layers 23 \
  --protect-heads 0,1,2,3,4,5,6 \
  --protect-max-delta 2 \
  --protect-qk-births 1 \
  --protect-vo-births 1 \
  --protect-vo-mode svd1 \
  --steps 160 \
  --batch-size 4 \
  --lr 2e-4 \
  --lora-r 8 \
  --grad-clip 0.3 \
  --adam-eps 1e-6 \
  --lambda-protect 10.0 \
  --protect-loss l1 \
  --lambda-qk 1.0 \
  --lambda-vo 1.0 \
  --log-every 20
```

## Multi-seed проверка

```bash
for S in 123 124 125; do
  python experiments/circuit_signal_birth_qwen/circuit_birth_finetune_protect_v1_3.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --device cuda \
    --dtype fp16 \
    --attn-implementation eager \
    --layers 23 \
    --protect-heads 0,1,2,3,4,5,6 \
    --protect-max-delta 2 \
    --protect-qk-births 1 \
    --protect-vo-births 1 \
    --protect-vo-mode svd1 \
    --steps 160 \
    --batch-size 4 \
    --lr 2e-4 \
    --lora-r 8 \
    --grad-clip 0.3 \
    --adam-eps 1e-6 \
    --lambda-protect 10.0 \
    --protect-loss l1 \
    --seed $S \
    --log-every 40
 done
```
