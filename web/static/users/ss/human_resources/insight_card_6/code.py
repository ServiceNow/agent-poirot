from agentpoirot.tools import func_tools
from agentpoirot.utils import save_stats_fig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/ss/human_resources/dataset.csv')

# Data Manipulation
# Create a new column 'turnover' to indicate if an employee has left the company
df['turnover'] = df['left_company']

# Plotting
# Set the style of seaborn
sns.set(style="whitegrid")

# Create a bar plot to compare turnover rates between remote and non-remote employees
plt.figure(figsize=(10, 6))
turnover_plot = sns.barplot(
    data=df,
    x='is_remote',
    y='turnover',
    estimator=lambda x: sum(x) / len(x),  # Calculate turnover rate
    ci=None,
    palette='muted'
)

# Set plot labels and title
turnover_plot.set_xticklabels(['Non-Remote', 'Remote'])
turnover_plot.set_xlabel('Work Mode')
turnover_plot.set_ylabel('Turnover Rate')
turnover_plot.set_title('Turnover Rates: Remote vs Non-Remote Employees')

# Save the plot and its stats
save_stats_fig(df, fig=turnover_plot.get_figure())