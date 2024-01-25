import torch

class adam_optimiser:
    def __init__(self, network, learning_rate=1e-2, weight_decay=0.0001, scheduler='simple_decay') -> None:
        self.schedule_type = scheduler
        self.optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), 
                                          lr=learning_rate, 
                                          weight_decay = weight_decay)
        
        if self.schedule_type =='cyclic':
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimiser, 
                                                          base_lr = 1e-3, 
                                                          max_lr = 1e-1, 
                                                          cycle_momentum=False, 
                                                          step_size_up=20, 
                                                          step_size_down=20)
        
    def scheduler_step(self, decay_rate = 0.95):
        if self.schedule_type =='simple_decay':
            for param_group in self.optimiser.param_groups:
                param_group['lr'] *= decay_rate
        else:
            self.scheduler.step()

    def get_lr(self):
        for param_group in self.optimiser.param_groups:
            return param_group['lr']
        
    def get_optimiser(self):
        return self.optimiser
    
    def zero_grad(self):
        self.optimiser.zero_grad()
    
    def step(self):
        self.optimiser.step()

    def state_dict(self):
        return self.optimiser.state_dict()