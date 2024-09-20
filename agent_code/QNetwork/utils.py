import torch
from .QNetwork import QNAgent

def setup(self):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure the state dimension matches the feature space (e.g., 350)
    self.agent = QNAgent(state_dim=350, action_dim=6, device=device)
    self.replay_buffer = self.agent.replay_buffer 
    self.q_network = self.agent.q_network
    self.device = device
    self.logger.info("QN agent setup completed.")
