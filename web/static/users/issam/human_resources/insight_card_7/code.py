from agentpoirot.tools import func_tools
from agentpoirot.utils import save_stats_fig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/issam/human_resources/dataset.csv')

# Data Manipulation
# Create a new column 'is_high_performer' based on 'last_performance_rating'
df['is_high_performer'] = df['last_performance_rating'] >= 4.0

# Calculate the average salary of high performers
df_high_performers = df[df['is_high_performer']]

# Plotting
# Set up the matplotlib figure
plt.figure(figsize=(8, 6))

# Create a bar plot for average salary of high performers
sns.barplot(x=['High Performers'], y=[df_high_performers['salary'].mean()], palette='viridis')

# Add labels and title
plt.title('Average Salary of High Performers')
plt.ylabel('Average Salary')
plt.xlabel('Performance Category')

# Save the plot and its stats
save_stats_fig(df, fig=plt.gcf())