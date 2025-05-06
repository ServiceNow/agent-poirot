from agentpoirot.tools import func_tools
import pandas as pd

# Load the dataframe
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/guest/itsm/dataset.csv')

# Data Manipulation
# Create a new column 'sla_breach_numeric' to convert 'sla_breach' to numeric for counting
df['sla_breach_numeric'] = df['sla_breach'].apply(lambda x: 1 if x == 'True' else 0)

# Group by 'assigned_group' and sum the 'sla_breach_numeric' to get the count of SLA breaches per group
df = df.groupby('assigned_group', as_index=False)['sla_breach_numeric'].sum()

# Plotting
# Use plot_bar to visualize the SLA breach counts per assigned group
func_tools.plot_bar(df=df, plot_column='assigned_group', count_column='sla_breach_numeric', plot_title='SLA Breaches by Assigned Group')