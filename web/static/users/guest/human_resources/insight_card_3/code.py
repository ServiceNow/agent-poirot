from agentpoirot.tools import func_tools
import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv('/mnt/home/projects/agent-poirot/web/static/users/guest/human_resources/dataset.csv')

# Stage Name: Data Manipulation
# Create a turnover column based on employment status
df['turnover'] = df['employment_status'].apply(lambda x: 1 if x == 'Terminated' else 0)

# Group by department and calculate turnover rate and average performance rating
df = df.groupby('department').agg(turnover_rate=('turnover', 'mean'), 
                                   average_performance_rating=('performance_rating', 'mean')).reset_index()

# Stage Name: Plotting
# Create a bar plot to visualize turnover rate by department and its correlation with performance ratings
func_tools.plot_bar(df=df, plot_column='department', count_column='turnover_rate', plot_title='Turnover Rate by Department')