# DistilGPT2 all-ideas result

Seed 0, conflict random_vocab, retain WikiText.

LoRA: patch_improve=+5.7580, forget_loss=+0.2864.
Full BP: patch_improve=+5.7647, forget_loss=+0.1725.
Magnitude sparse: patch_improve=+5.7294, forget_loss=-0.0875.
Error-Wave v4: patch_improve=+5.7068, forget_loss=-0.2062.
CGA hard: patch_improve=+5.5714, forget_loss=-1.2765.

Вывод: Error-Wave v4 почти достигает LoRA/full_bp по patch, но не ухудшает retain. CGA hard — safe-mode: patch ниже, retain намного лучше.

DRCF пока не дал отдельного эффекта: drcf_v4 ~= error_wave_v4, drcf_cga ~= cga_hard.
