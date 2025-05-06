from agentpoirot.tools import func_tools
from agentpoirot.utils import save_stats_fig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/ss/human_resources/dataset.csv')

# Stage 1: Data Manipulation
# Calculate turnover rate for each job level
df['turnover'] = df['left_company']  # Assuming 'left_company' is 1 if left, 0 if not
df = df.groupby('job_level').agg(turnover_rate=('turnover', 'mean')).reset_index()

# Stage 2: Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='job_level', y='turnover_rate', palette='viridis')
plt.title('Turnover Rate Across Different Job Levels')
plt.xlabel('Job Level')
plt.ylabel('Turnover Rate')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot and its stats
save_stats_fig(df, fig=plt.gcf())