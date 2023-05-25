import wandb

def run(config):
    with wandb.init(config=config, settings={"disable_git": True}):
        config = wandb.config
        for epoch in range(1, config.epochs):
            loss = config.epochs / epoch
            accuracy = (1 + (epoch / config.epochs))/2
            wandb.log({
                "loss": loss,
                "accuracy": accuracy,
                "epoch": epoch})
        wandb.run.log_code() 