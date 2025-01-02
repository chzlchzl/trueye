from torch.utils.tensorboard import SummaryWriter  # PyTorch의 SummaryWriter 사용

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)  # FileWriter 대신 SummaryWriter 사용
        print(f"TensorBoard logging directory: {log_dir}")

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def close(self):
        """Close the summary writer."""
        self.writer.close()
