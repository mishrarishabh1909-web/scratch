import yaml
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from fog_rl_medical.training.trainer import Trainer

def main():
    print("Loading config...")
    try:
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Load the actual config files
        config = {}
        for key, path in cfg.get('configs', {}).items():
            try:
                with open(path, 'r') as f:
                    config[key] = yaml.safe_load(f)
            except FileNotFoundError:
                print(f"Warning: Config file {path} not found, using defaults")
                config[key] = {}
    except FileNotFoundError:
        config = {}
        
    trainer = Trainer(config)
    trainer.run()

if __name__ == "__main__":
    main()
