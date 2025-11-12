# SOEN472
MLP model:
1.
* Created and activated a clean isolated Python environment.
* Installed all necessary data science and ML dependencies.
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision scikit-learn numpy pandas matplotlib tqdm seaborn

2. Recreate my exact setup with:
pip install -r requirements.txt

3. Run python scripts/make_csv_features.py
4. Do : source venv/bin/activate
5. Run: python mlp_model.py

MLP files:
mlp_mode.py : contains the MLP implementation with different variants (depths and layers) and creates a confusion matrix for each model based on the base model. 
mlp_depth_summary (layer variants): contains the model name, Accuracy, Precision, Recall, F1 for each 1 layer and 2 layer mlp models.
mlp_variants_summary (size variants): contains the model name, Accuracy, Precision, Recall, F1 for each narrow, wide and extra wide mlp models.
requirements.txt: recreate the python environmnent to test the mlp model.
mlp_1layer.pth, mlp_3layer.pth, mlp_narrow.pth, mlp_wide.pth, mlp_extra_wide.pth, mlp_base.pth: are the saved models




