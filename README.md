
ADVERSARIAL TRAINING + AUTOATTACK EVALUATION

Needed to install
-------------------------
pip install torch torchvision
pip install git+https://github.com/fra31/auto-attack

models.py (or a models/ folder) with ResNet18 in it is already included


HOW TO RUN TRAINING
-------------------

Run a single script:
    python qadv_trades_untarget.py my_experiment



This trains for 110 epochs and saves:
- Checkpoints → ./Checkpoints/my_experiment/qadv_trades_untarget/
- Results (CSVs) → ./CSV/my_experiment/qadv_trades_untarget/

Checkpoints start at epoch 71 and go to epoch 110 (epoch_72_model.pth, etc.)



HOW TO RUN AUTOATTACK EVALUATION
---------------------------------

1. Find where your checkpoints are saved
   Example: ./Checkpoints/my_experiment/qadv_trades_untarget/

2. Open eval_AA.py and edit these two lines:

   checkpoint_base = './Checkpoints/my_experiment/qadv_trades_untarget'

3. Run it:
   python eval_AA.py

4. Results saved in ./AutoAttack_Results_Epochs/
   Check AutoAttack_Summary_Compact.csv for quick overview


WHAT THE SCRIPTS DO
--------------------
- qadv_pgd_untarget.py : Q-Advantage with PGD warmup and untargeted attacks
- qadv_pgd_target.py : Q-Advantage with PGD warmup and targeted attacks  
- qadv_trades_untarget.py : Q-Advantage with TRADES warmup and untargeted attacks
- qadv_trades_target.py : Q-Advantage with TRADES warmup and targeted attacks
- qadv_trades_conf.py : Q-Advantage with TRADES warmup and confusion matrix adjusted attack

- eval_AA.py : AutoAttack evaluation (runs on saved checkpoints)


WORKFLOW
----------------
1. python qadv_trades_untarget.py my_exp
3. Edit eval_AA.py:
   - checkpoint_base = './Checkpoints/my_exp/qadv_trades_untarget'
4. python eval_AA.py
5. Check AutoAttack_Results_Epochs/AutoAttack_Summary_Compact.csv
================================================================================
