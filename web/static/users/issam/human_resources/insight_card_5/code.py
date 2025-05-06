from agentpoirot.tools import func_tools
from agentpoirot.utils import save_stats_fig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/issam/human_resources/dataset.csv')

# Stage Name: Data Manipulation
# Create a new column to indicate if an employee received a promotion in the last two years
df['received_promotion'] = df['promotion_last_2_years'] == 1.0

# Calculate turnover rates for employees who received a promotion and those who did not
df_turnover = df.groupby('received_promotion')['left_company'].mean().reset_index()

# Stage Name: Plotting
# Plot the turnover rates based on promotion status
plt.figure(figsize=(8, 6))
sns.barplot(data=df_turnover, x='received_promotion', y='left_company', palette='viridis')
plt.title('Turnover Rates Based on Promotion Status')
plt.xlabel('Received Promotion in Last 2 Years')
plt.ylabel('Turnover Rate')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.ylim(0, 1)

# Save the plot and its stats
save_stats_fig(df_turnover, fig=plt.gcf())