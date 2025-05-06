from agentpoirot.tools import func_tools
from agentpoirot.utils import save_stats_fig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/issam/human_resources/dataset.csv')

# Data Manipulation
# Create a new column to categorize employees based on their performance and promotion status
df['performance_promotion_category'] = df.apply(
    lambda row: 'High Performer & Promoted' if row['last_performance_rating'] >= 4 and row['promotion_last_2_years'] == 1 else
                'High Performer & Not Promoted' if row['last_performance_rating'] >= 4 else
                'Low Performer & Promoted' if row['last_performance_rating'] < 4 and row['promotion_last_2_years'] == 1 else
                'Low Performer & Not Promoted',
    axis=1
)

# Plotting
# Create a count plot to visualize the distribution of performance and promotion categories
plt.figure(figsize=(10, 6))
sns.set_palette("pastel")
ax = sns.countplot(data=df, x='performance_promotion_category', order=[
    'High Performer & Promoted', 
    'High Performer & Not Promoted', 
    'Low Performer & Promoted', 
    'Low Performer & Not Promoted'
])
ax.set_title('Distribution of Performance and Promotion Categories')
ax.set_xlabel('Performance and Promotion Category')
ax.set_ylabel('Number of Employees')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot and its stats
save_stats_fig(df, fig=ax.get_figure())