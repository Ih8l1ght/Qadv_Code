
ADVERSARIAL TRAINING + AUTOATTACK EVALUATION

Needed to install
-------------------------
pip install torch torchvision
pip install git+https://github.com/fra31/auto-attack

models.py (or a models/ folder) with ResNet18 in it is already included


USAGE:

    python main.py my_experiment_name

This will:
1. Run qadv_trades_untarget.py (can be modified inside main.py) with "my_experiment_name"
2. Wait for training to complete (110 epochs)
3. Automatically run AutoAttack evaluation on selected checkpoints
4. Save all results in organized directories



WHAT THE SCRIPTS DO
--------------------
- qadv_pgd_untarget.py : Q-Advantage with PGD warmup and untargeted attacks
- qadv_pgd_target.py : Q-Advantage with PGD warmup and targeted attacks  
- qadv_trades_untarget.py : Q-Advantage with TRADES warmup and untargeted attacks
- qadv_trades_target.py : Q-Advantage with TRADES warmup and targeted attacks
- qadv_trades_conf.py : Q-Advantage with TRADES warmup and confusion matrix adjusted attack

- eval_AA.py : AutoAttack evaluation (runs on saved checkpoints)
- main.py : Runs both the training and evaluation in the same folder

