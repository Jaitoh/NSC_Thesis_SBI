"""

"""

class Solver:
    def __init__(self, config, model, train_loader, val_loader, test_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.best_val_loss = float('inf')
        self.best_test_loss = float('inf')

        self._build()