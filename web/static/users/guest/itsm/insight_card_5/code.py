from agentpoirot.tools import func_tools
import pandas as pd

# Load the dataframe
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/guest/itsm/dataset.csv')

# Data Manipulation
# Create a new column 'impact_priority' to combine 'impact' and 'priority' for correlation analysis
df['impact_priority'] = df['impact'] + '_' + df['priority']

# Plotting
# Use plot_bar to visualize the correlation between impact and priority
func_tools.plot_bar(df=df, plot_column='impact_priority', count_column='ticket_id', plot_title='Correlation between Impact and Priority of Tickets')