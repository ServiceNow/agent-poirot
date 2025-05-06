from agentpoirot.tools import func_tools
from agentpoirot.utils import save_stats_fig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/issam/human_resources/dataset.csv')

# Stage 1: Data Manipulation
# Add a column to categorize employees based on their performance rating
df['performance_category'] = pd.cut(df['last_performance_rating'], 
                                    bins=[0, 2, 3, 4, 5], 
                                    labels=['Low', 'Average', 'Good', 'Excellent'])

# Stage 2: Plotting
# Create a count plot to show the distribution of employees across performance categories
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='performance_category', palette='viridis')
plt.title('Distribution of Employees by Performance Category')
plt.xlabel('Performance Category')
plt.ylabel('Number of Employees')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot and its stats
extra_stats = {
    'total_employees': len(df),
    'average_performance_rating': df['last_performance_rating'].mean()
}
save_stats_fig(df, extra_stats=extra_stats, fig=plt.gcf())