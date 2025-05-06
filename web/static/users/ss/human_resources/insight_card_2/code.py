from agentpoirot.tools import func_tools
from agentpoirot.utils import save_stats_fig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/ss/human_resources/dataset.csv')

# Data Manipulation
# Create a new column 'turnover' based on 'left_company' column
df['turnover'] = df['left_company'].apply(lambda x: 'Left' if x == 1 else 'Stayed')

# Plotting
# Create a boxplot to visualize the relationship between salary and employee turnover
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='turnover', y='salary', palette='Set2')
plt.title('Relationship between Salary and Employee Turnover')
plt.xlabel('Employee Turnover')
plt.ylabel('Salary')
plt.xticks(rotation=45)

# Save the plot and its stats
save_stats_fig(df, fig=plt.gcf())