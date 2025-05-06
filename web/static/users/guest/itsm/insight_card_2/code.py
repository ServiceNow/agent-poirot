from agentpoirot.tools import func_tools
import pandas as pd

# Load the dataframe
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/guest/itsm/dataset.csv')

# Data Manipulation
# Create a new column 'requester_request_type' to combine requester and request_type
df['requester_request_type'] = df['requester'] + ' - ' + df['request_type']

# Create a count column to count occurrences of each requester_request_type
df['count'] = 1

# Plotting
# Use plot_bar to visualize the frequency of each requester_request_type
func_tools.plot_bar(df=df, plot_column='requester_request_type', count_column='count', plot_title='Frequency of Request Types by Requester')