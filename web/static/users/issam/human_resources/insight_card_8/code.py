from agentpoirot.tools import func_tools
from agentpoirot.utils import save_stats_fig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/issam/human_resources/dataset.csv')

# Data Manipulation
# Create a new column 'high_performer' to identify high performers with a rating of 4 or 5
df['high_performer'] = df['last_performance_rating'] >= 4

# Plotting
# Set the style of seaborn
sns.set(style="whitegrid")

# Create a violin plot to show the salary distribution among high performers
plt.figure(figsize=(10, 6))
sns.violinplot(x='high_performer', y='salary', data=df, palette='muted')
plt.title('Salary Distribution Among High Performers')
plt.xlabel('High Performer')
plt.ylabel('Salary')

# Save the plot and its stats
save_stats_fig(df, fig=plt.gcf())