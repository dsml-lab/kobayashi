import wandb

def list_wandb_runs(entity='dsml-kernel24', project='kobayashi_save_model'):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    
    print(f"Listing available run names in '{entity}/{project}':")
    for r in runs:
        print(r.name)

if __name__ == "__main__":
    list_wandb_runs()