from agentpoirot.tools import func_tools
from agentpoirot.utils import save_stats_fig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/issam/human_resources/dataset.csv')

# Data Manipulation
# Create a new column to indicate if an employee has left the company
df['left_company'] = df['left_company'].astype(int)

# Group by location and calculate the turnover rate
df = df.groupby('location').agg(turnover_rate=('left_company', 'mean')).reset_index()

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='location', y='turnover_rate', palette='viridis')
plt.title('Employee Turnover Rate by Location')
plt.xlabel('Location')
plt.ylabel('Turnover Rate')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot and stats
save_stats_fig(df, extra_stats=None, fig=plt.gcf())