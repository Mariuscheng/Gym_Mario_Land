import numpy as np
import time, datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.rcsetup import interactive_bk, non_interactive_bk
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib as mpl
from pathlib import Path

matplotlib.backends.backend_registry.list_builtin(matplotlib.backends.BackendFilter.INTERACTIVE)
class MarioMetricLogger:
    def __init__(self, save_dir, resume=False):
        self.resume = resume
        self.save_log = save_dir / "log.txt"
        self.save_dir = save_dir
        self.load_log()

        # Set up modern style configuration
        plt.style.use('seaborn-v0_8-darkgrid')  # Updated style name
        
        # Configure default plotting parameters
        mpl.rcParams.update({
            'figure.figsize': (10, 6),
            'lines.linewidth': 2,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'grid.linestyle': '--',
            'grid.alpha': 0.7
        })

        # Plot file paths
        self.ep_rewards_plot = save_dir / "reward_plot.png"  # Changed to PNG for better quality
        self.ep_lengths_plot = save_dir / "length_plot.png"
        self.ep_avg_losses_plot = save_dir / "loss_plot.png"
        self.ep_avg_qs_plot = save_dir / "q_plot.png"

        # Timing
        self.record_time = time.time()

    def load_log(self):
        if self.save_log.exists() and self.resume:
            data = np.loadtxt(self.save_log, skiprows=1, unpack=True)
            if data.size > 0:
                self.episode, self.step, self.epsilon, self.ep_rewards, self.ep_lengths, self.ep_avg_losses, self.ep_avg_qs = data
                if isinstance(self.episode,list):
                    self.episode = list(self.episode.astype(int))
                    self.step = list(self.step.astype(int))
                    self.ep_rewards = list(self.ep_rewards)
                    self.ep_lengths = list(self.ep_lengths)
                    self.ep_avg_losses = list(self.ep_avg_losses)
                    self.ep_avg_qs = list(self.ep_avg_qs)
                else:
                    self.episode = [self.episode]
                    self.step = [self.step]
                    self.ep_rewards = list(self.ep_rewards)
                    self.ep_lengths = list(self.ep_lengths)
                    self.ep_avg_losses = list(self.ep_avg_losses)
                    self.ep_avg_qs = list(self.ep_avg_qs)
            else:  # Handle empty log file with header
                self.reset_lists()
        else:
            self.reset_lists()
            with open(self.save_log, "w") as f:
                f.write(
                    "Episode    Step    Epsilon    MeanReward    MeanLength    MeanLoss    MeanQValue\n"
                )
        self.init_episode()

    def reset_lists(self):
        self.episode = []
        self.step = []
        self.epsilon = []
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        self.init_episode()

    def record(self, episode, epsilon, step):
        #print("shape: ",len(self.ep_rewards))
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_delta = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon:.5f} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_delta} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:<11}{step:<15}{epsilon:<12.5f}"
                f"{mean_ep_reward:<29.0f}{mean_ep_length:<16.0f}{mean_ep_loss:<15.0f}{mean_ep_q:<15.0f}\n"
            )
        self.plot_metrics()

    def plot_metrics(self):
        metrics = [
            ("ep_rewards", "Reward", "tab:blue"),
            ("ep_lengths", "Length", "tab:green"),
            ("ep_avg_losses", "Loss", "tab:red"),
            ("ep_avg_qs", "Q Value", "tab:purple")
        ]
        
        for metric, name, color in metrics:
            # Create a new figure using the modern object-oriented interface
            fig = Figure(figsize=(10, 6), dpi=100)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            
            data = getattr(self, metric)
            if not data:  # Skip if no data
                continue
                
            # Plot raw data with alpha
            ax.plot(data, alpha=0.3, color=color, label=f"Raw {name}", 
                   linewidth=1, zorder=1)
            
            # Plot moving average with enhanced styling
            ma = self.calculate_moving_average(data)
            ax.plot(ma, color=color, label=f"Moving Avg {name}",
                   linewidth=2.5, zorder=2)
            
            # Enhanced styling
            ax.set_title(f"{name} over Episodes", pad=15)
            ax.set_xlabel("Episode")
            ax.set_ylabel(name)
            
            # Add grid with modern styling
            ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
            
            # Enhance legend
            ax.legend(frameon=True, fancybox=True, shadow=True,
                     loc='upper left', bbox_to_anchor=(1, 1))
            
            # Adjust layout to prevent label cutoff
            fig.tight_layout()
            
            # Save with high quality and proper bbox
            canvas.print_figure(self.save_dir / f"{metric}_plot.png",
                              bbox_inches='tight',
                              pad_inches=0.1,
                              facecolor='white',
                              edgecolor='none')
            
            # Clean up
            plt.close(fig)

    def calculate_moving_average(self, data, window_size=100):
        """
        Calculate moving average using convolution for better performance.
        
        Args:
            data (list): Input data
            window_size (int): Size of the moving window
            
        Returns:
            numpy.ndarray: Moving average of the input data
        """
        if not data:
            return []
            
        data_arr = np.array(data)
        if len(data_arr) < window_size:
            window_size = len(data_arr)
        if window_size < 1:
            return data_arr
            
        # Use convolution for efficient moving average
        window = np.ones(window_size) / window_size
        return np.convolve(data_arr, window, mode='valid')