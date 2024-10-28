from agentpoirot.tools import func_tools
import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv('/mnt/home/projects/agent-poirot/web/static/users/guest/human_resources/dataset.csv')

# Data Manipulation
# Create a turnover column based on employment status
df['turnover'] = df['employment_status'].apply(lambda x: 1 if x == 'Terminated' else 0)

# Group by department and calculate turnover rate and average performance rating
df['total_employees'] = 1  # Create a column to count total employees
turnover_stats = df.groupby('department').agg(turnover_rate=('turnover', 'mean'), 
                                                avg_performance_rating=('performance_rating', 'mean')).reset_index()

# Plotting
# Create a bar plot to visualize turnover rate by department and its correlation with performance ratings
func_tools.plot_bar(df=turnover_stats, plot_column='department', count_column='turnover_rate', plot_title='Turnover Rate by Department')