# Import necessary libraries
import pandas as pd
from agentpoirot.tools import func_tools

# Load the dataset into a DataFrame
df = pd.read_csv('/mnt/home/projects/agent-poirot/web/static/users/guest/human_resources/dataset.csv')

# Stage 1: Data Manipulation
# Group by department and calculate the average performance rating
df = df.groupby('department', as_index=False).agg({'performance_rating': 'mean'})

# Stage 2: Plotting
# Create a bar plot for average performance rating by department
func_tools.plot_bar(df=df, plot_column='department', count_column='performance_rating', plot_title='Average Performance Rating by Department')