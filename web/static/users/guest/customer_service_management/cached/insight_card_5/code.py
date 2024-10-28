from agentpoirot.tools import func_tools
import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv('/mnt/home/projects/research-skilled-poirot/web/static/users/issam.laradji@servicenow.com/customer_service_management/dataset.csv')

# Data Manipulation
# Calculate the average time taken to close cases
df['close_duration'] = (pd.to_datetime(df['closed_at']) - pd.to_datetime(df['resolved_at'])).dt.total_seconds() / 3600  # in hours
# Calculate the average resolution time
df['resolution_duration'] = df['rpt_case_resolve_duration']  # already in hours

# Plotting
# Create a box plot to compare average close duration and resolution duration
func_tools.plot_boxplot(df=df, x_column='state', y_column='close_duration', plot_title='Comparison of Average Close Duration and Resolution Duration')