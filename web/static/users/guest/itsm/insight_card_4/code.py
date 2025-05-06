from agentpoirot.tools import func_tools
import pandas as pd

# Load the dataframe
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/guest/itsm/dataset.csv')

# Data Manipulation
# Create a new column 'request_category' to combine 'request_type' and 'category'
df['request_category'] = df['request_type'] + ' - ' + df['category']
# Create a count column for plotting
df['count'] = 1

# Plotting
# Use plot_bar to visualize the correlation between request types and categories
func_tools.plot_bar(df=df, plot_column='request_category', count_column='count', plot_title='Correlation between Request Types and Categories')