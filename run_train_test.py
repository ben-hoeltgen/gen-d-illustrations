import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import os
import pickle
from pathlib import Path
import time
warnings.filterwarnings('ignore')

# Configuration
STATES = ['NY' 'TX', 'FL', 'CA']  # Four biggest states
TASKS = {
    'income': ACSIncome
}
MODELS = {
    'LR': LogisticRegression(random_state=0, max_iter=1000),
    'GBM': GradientBoostingClassifier(random_state=0), # n_estimators=100, max_depth=3, max_leaf_nodes=None, min_samples_leaf=1
    'GBM_opt': GradientBoostingClassifier(random_state=0, n_estimators=500, max_depth=10, max_leaf_nodes=50, min_samples_leaf=500), 
    'GBM_tun': GradientBoostingClassifier(random_state=0, n_estimators=400, max_depth=8, max_leaf_nodes=50, min_samples_leaf=500) 
}
TRAIN_SIZE = 0.5
N_SEEDS = 10
YEARS = [2014, 2015, 2016, 2017, 2018]


# Create output directory
OUTPUT_DIR = Path('folktables_results')
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data(state, year, task_def):
    """Load folktables data for a given state, year, and task."""
    data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    data = data_source.get_data(states=[state], download=True)
    X, y, _ = task_def.df_to_numpy(data)
    return X, y

def get_result_filename(exp_type, state, task_name, model_name):
    """Generate filename for saving results."""
    return OUTPUT_DIR / f"{exp_type}_{state}_{task_name}_{model_name}.csv"

def get_multiplicity_filename(state, task_name, model_name):
    """Generate filename for saving multiplicity data."""
    return OUTPUT_DIR / f"mult_{state}_{task_name}_{model_name}.pkl"

def check_if_done(exp_type, state, task_name, model_name):
    """Check if this experiment has already been completed."""
    filename = get_result_filename(exp_type, state, task_name, model_name)
    return filename.exists()

def experiment_a_fixed_training(state, task_name, task_def, model_name, model_class):
    """
    Experiment A: Fixed training year (2014), evaluate on 2014-2018.
    """
    # Check if already done
    if check_if_done('exp_a', state, task_name, model_name):
        print(f"    Experiment A already completed, skipping...")
        filename = get_result_filename('exp_a', state, task_name, model_name)
        return pd.read_csv(filename)
    
    results = []
    
    # Load 2014 training data
    X_full, y_full = load_data(state, 2014, task_def)
    
    # Run with different seeds
    for seed in range(N_SEEDS):
        X_train, _, y_train, _ = train_test_split(
            X_full, y_full, train_size=TRAIN_SIZE, random_state=seed
        )
        
        # Train model
        model = model_class.__class__(**{k: v for k, v in model_class.get_params().items()})
        model.set_params(random_state=seed)
        model.fit(X_train, y_train)
        
        # Evaluate on each year
        for eval_year in YEARS:
            X_eval, y_eval = load_data(state, eval_year, task_def)
            _, X_test, _, y_test = train_test_split(
                X_eval, y_eval, train_size=TRAIN_SIZE, random_state=seed
            )
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            results.append({
                'state': state,
                'task': task_name,
                'model': model_name,
                'seed': seed,
                'train_year': 2014,
                'eval_year': eval_year,
                'accuracy': acc
            })
    
    df = pd.DataFrame(results)
    
    # Save immediately
    filename = get_result_filename('exp_a', state, task_name, model_name)
    df.to_csv(filename, index=False)
    print(f"    Saved: {filename}")
    
    return df

def experiment_b_fixed_test(state, task_name, task_def, model_name, model_class):
    """
    Experiment B: Fixed test year (2018), train on different year combinations.
    """
    # Check if already done
    if check_if_done('exp_b', state, task_name, model_name):
        print(f"    Experiment B already completed, skipping...")
        filename = get_result_filename('exp_b', state, task_name, model_name)
        mult_filename = get_multiplicity_filename(state, task_name, model_name)
        df = pd.read_csv(filename)
        with open(mult_filename, 'rb') as f:
            multiplicity_data = pickle.load(f)
        return df, multiplicity_data
    
    results = []
    multiplicity_data = {}
    
    # Load 2018 test data once
    X_test_2018, y_test_2018 = load_data(state, 2018, task_def)
    
    # Define training year combinations
    train_configs = [
        ([2018], '2018'),
        ([2017], '2017'),
        ([2014, 2015, 2016, 2017], '2014-2017')
    ]
    
    for seed in range(N_SEEDS):
        seed_probs = {}
        
        for train_years, config_name in train_configs:
            # Load and combine training data
            X_list, y_list = [], []
            for year in train_years:
                X_year, y_year = load_data(state, year, task_def)
                # Split training data
                X_year_train, _, y_year_train, _ = train_test_split(
                    X_year, y_year, train_size=TRAIN_SIZE, random_state=seed
                )
                X_list.append(X_year_train)
                y_list.append(y_year_train)
            
            X_train = np.vstack(X_list)
            y_train = np.hstack(y_list)
            
            
            
            # Train model
            model = model_class.__class__(**{k: v for k, v in model_class.get_params().items()})
            model.set_params(random_state=seed)
            model.fit(X_train, y_train)
            
            # Evaluate on 2018
            y_pred = model.predict(X_test_2018)
            y_prob = model.predict_proba(X_test_2018)[:, 1]
            acc = accuracy_score(y_test_2018, y_pred)
            
            results.append({
                'state': state,
                'task': task_name,
                'model': model_name,
                'seed': seed,
                'train_config': config_name,
                'train_years': '-'.join(map(str, train_years)),
                'eval_year': 2018,
                'accuracy': acc
            })
            
            # Store probabilities for multiplicity calculation
            if config_name not in seed_probs:
                seed_probs[config_name] = []
            seed_probs[config_name].append(y_prob)
        
        # Store probabilities for this seed
        for config_name, probs in seed_probs.items():
            if config_name not in multiplicity_data:
                multiplicity_data[config_name] = []
            multiplicity_data[config_name].extend(probs)
    
    df = pd.DataFrame(results)
    
    # Save results immediately
    filename = get_result_filename('exp_b', state, task_name, model_name)
    df.to_csv(filename, index=False)
    print(f"    Saved: {filename}")
    
    # Save multiplicity data
    mult_filename = get_multiplicity_filename(state, task_name, model_name)
    with open(mult_filename, 'wb') as f:
        pickle.dump(multiplicity_data, f)
    print(f"    Saved: {mult_filename}")
    
    return df, multiplicity_data


def main():
    """Run all experiments."""
    
    total_configs = len(STATES) * len(TASKS) * len(MODELS)
    current = 0
    
    print("="*80)
    print(f"Starting experiments. Results will be saved to: {OUTPUT_DIR}/")
    print("Progress can be resumed if interrupted.")
    print("="*80 + "\n")
    
    for state in STATES:
        for task_name, task_def in TASKS.items():
            for model_name, model_class in MODELS.items():
                current += 1
                print(f"\n[{current}/{total_configs}] Processing: {state}, {task_name}, {model_name}")
                
                try:
                    # Experiment A
                    print("  Running Experiment A (fixed training year)...")
                    start_time = time.time()
                    df_a = experiment_a_fixed_training(state, task_name, task_def, model_name, model_class)
                    print(f"    Experiment A completed in {(time.time() - start_time)/60:.1f}min")
                    
                    # Experiment B
                    print("  Running Experiment B (fixed test year)...")
                    start_time = time.time()
                    df_b, df_mult = experiment_b_fixed_test(state, task_name, task_def, model_name, model_class)
                    print(f"    Experiment B completed in {(time.time() - start_time)/60:.1f}min")
                    
                    print(f"  ✓ Completed successfully")
                    
                except Exception as e:
                    print(f"  ✗ Error occurred: {e}")
                    print(f"  Results saved so far can be loaded later.")
                    continue
    
    print("\n" + "="*80)
    print("Loading and combining all results...")
    print("="*80)
    
    print("\n" + "="*80)
    print(f"All results saved to: {OUTPUT_DIR}/")
    print("  - exp_a_[state]_[task]_[model].csv")
    print("  - exp_b_[state]_[task]_[model].csv")
    print("  - mult_[state]_[task]_[model].pkl")
    
    return

if __name__ == "__main__":
    main()