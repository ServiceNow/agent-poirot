from agentpoirot.tools import func_tools
from agentpoirot.utils import save_stats_fig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/ss/human_resources/dataset.csv')

# Data Manipulation
# Create a new column to indicate if an employee received a promotion in the last two years
df['received_promotion'] = df['promotion_last_2_years'] == 1.0

# Plotting
# Set the style of seaborn
sns.set(style="whitegrid")

# Create a bar plot to show the effect of receiving a promotion on turnover
plt.figure(figsize=(10, 6))
turnover_plot = sns.barplot(
    data=df,
    x='received_promotion',
    y='left_company',
    estimator=lambda x: sum(x) / len(x),  # Calculate turnover rate
    ci=None,
    palette='pastel'
)

# Set plot labels and title
turnover_plot.set_xlabel('Received Promotion in Last 2 Years', fontsize=12)
turnover_plot.set_ylabel('Turnover Rate', fontsize=12)
turnover_plot.set_title('Effect of Promotion on Employee Turnover', fontsize=14)
turnover_plot.set_xticklabels(['No', 'Yes'])

# Save the plot and its stats
save_stats_fig(df, fig=turnover_plot.get_figure())