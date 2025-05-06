from agentpoirot.tools import func_tools
from agentpoirot.utils import save_stats_fig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/issam/human_resources/dataset.csv')

# Data Manipulation
# Create a new column to indicate if an employee has left the company
df['has_left'] = df['left_company']

# Group by gender and calculate the turnover rate
df = df.groupby('gender').agg(turnover_rate=('has_left', 'mean')).reset_index()

# Plotting
# Set the style of the plot
sns.set(style="whitegrid")

# Create a bar plot to visualize the turnover rates by gender
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(data=df, x='gender', y='turnover_rate', palette='viridis')

# Set plot labels and title
bar_plot.set_title('Employee Turnover Rates by Gender')
bar_plot.set_xlabel('Gender')
bar_plot.set_ylabel('Turnover Rate')

# Save the plot and its statistics
save_stats_fig(df, fig=bar_plot.get_figure())