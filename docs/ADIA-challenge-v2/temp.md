================================================================================
      RUNNING TIER 3: FULL END-TO-END TRAINING
================================================================================
This will take a long time, but only needs to be run once to generate
the 'pretrained_autoencoder.pth' file for faster iteration.

Creating a large mock dataset...
Training set size: 1000, Test set size: 200

✅ Found and using GPU.

--- Starting Training Pipeline ---
--- Starting Stage 1: Pre-training ---
Generating Base ECAs:   0%|          | 0/28 [00:00<?, ?it/s]
Generating Composite ECAs: 0it [00:00, ?it/s]
Pre-train Epoch 1/5:   0%|          | 0/88 [00:00<?, ?it/s]

ERROR during full run: Subtraction, the `-` operator, with two bool tensors is not supported. Use the `^` or `logical_xor()` operator instead.
Traceback (most recent call last):
  File "/tmp/ipykernel_13563/4175275179.py", line 26, in <module>
    train(train_X, train_y, config.MODEL_DIR)
  File "/tmp/ipykernel_13563/2815794562.py", line 204, in train
    loss_recon = recon_loss_fn(recon_logits, sequences > 0.5); loss_rule = rule_loss_fn(rule_logits, labels)
  File "/home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/torch/nn/modules/loss.py", line 725, in forward
    return F.binary_cross_entropy_with_logits(input, target,
  File "/home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/torch/nn/functional.py", line 3199, in binary_cross_entropy_with_logits
    return torch.binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction_enum)
RuntimeError: Subtraction, the `-` operator, with two bool tensors is not supported. Use the `^` or `logical_xor()` operator instead.
