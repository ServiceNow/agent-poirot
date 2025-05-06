from agentpoirot.tools import func_tools
from agentpoirot.utils import save_stats_fig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/ss/human_resources/dataset.csv')

# Data Manipulation
# Calculate turnover rate by performance rating
df['turnover'] = df['left_company']
df_turnover_rate = df.groupby('last_performance_rating')['turnover'].mean().reset_index()

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=df_turnover_rate, x='last_performance_rating', y='turnover', palette='viridis')
plt.title('Turnover Rate by Performance Rating')
plt.xlabel('Performance Rating')
plt.ylabel('Turnover Rate')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot and its stats
save_stats_fig(df_turnover_rate, fig=plt.gcf())