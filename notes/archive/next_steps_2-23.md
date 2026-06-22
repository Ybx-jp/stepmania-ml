● Based on the training curves, evaluation results, and project history, here are the most impactful next steps, ordered by expected payoff:
                                                                                                                                                                                        
  1. Train longer (easy win)                                                                                                                                                            
                                                                                                                                                                                        
  The model is still improving at epoch 5 — train loss dropping, val loss dropping, no plateau yet. The config has early_stopping_patience: 5 and the scheduler hasn't even reduced LR. 
  Training for 20-30 epochs would likely push accuracy into the mid-80s just from convergence.                                                                                          
                                                                                                                                                                                        
  2. Address the Medium/Hard confusion                                                                                                                                                  
                                                                                                                                                                                        
  This is the biggest weakness. Hard recall is only 55% — nearly half of Hard charts get classified as Medium. Options:                                                                 
  - More data in the Hard class — Hard has the fewest samples and the boundary with Medium is fuzzy                                                                                     
  - Ordinal-aware loss — the project already has src/losses/ordinal.py and a DualTaskClassifier. Since difficulty is ordinal, penalizing Beginner-as-Hard more than Medium-as-Hard makes
   sense                                                                                                                                                                                
  - Groove radar features — these capture chart complexity dimensions (stream, voltage, chaos) that might discriminate Medium vs Hard better than raw patterns alone. The model supports
   use_groove_radar: true but it's worth verifying they're being utilized effectively                                                                                                   
                                                                                                                                                                                        
  3. More training data                                                                                                                                                                 
                                                                                                                                                                                        
  976 chart files yielding ~3200 samples is modest. The data/ directory has community/ alongside external/ — if there are more packs available, adding them would help, especially for  
  underrepresented classes.                                                                                                                                                             
                                                                                                                                                                                        
  4. Contrastive fine-tuning                                                                                                                                                            
                                                                                                                                                                                        
  The project has a fully built ContrastiveTrainer and two planned experiments (A: aggressive, B: conservative) that were never run to completion. Now that you have a strong 80%
  baseline, running Experiment A (freeze classifier head for 5 epochs, contrastive-only, then unfreeze) could reshape the embedding space to better capture groove similarity,
  potentially improving the Medium/Hard boundary.

  5. Hyperparameter tuning

  The project has an Optuna config (config/optuna_config.yaml) already set up. Key knobs to search:
  - Learning rate (currently 1e-4)
  - Backbone depth (currently 4 blocks, could try 6-8)
  - Dropout rates (backbone at 0.4 may be too aggressive for this data size)
  - Batch size (128 with accumulation_steps=2 gives effective 256)

  6. Data augmentation

  No augmentation is currently used. For chart data:
  - Mirror — flip left/right arrows (symmetric difficulty)
  - Temporal jitter — small shifts in note timing
  - Audio augmentation — pitch shift, time stretch

  ---
  My recommendation: start with #1 (just set num_epochs: 25) since it's zero-effort and the model is clearly still learning. Then tackle #2 (ordinal loss) since the error analysis
  shows 98% of errors are adjacent — exactly what ordinal losses are designed for.
