from agentpoirot.tools import func_tools
import pandas as pd

# Load the dataset
df = pd.read_csv('/mnt/home/projects/research-skilled-poirot/web/static/users/issam.laradji@servicenow.com/customer_service_management/dataset.csv')

# Data Manipulation
# Create a new column to indicate SLA breaches
df['sla_breach'] = df['made_sla'] == 0  # 0 indicates SLA was not met
# Group by impact level and count the number of SLA breaches
df['sla_breach_count'] = df.groupby('impact')['sla_breach'].transform('sum')  # Count of breaches per impact level
# Subset the dataframe to include only necessary columns for plotting
df = df[['impact', 'sla_breach_count']].drop_duplicates()  # Keep unique combinations of impact and breach count

# Plotting
func_tools.plot_bar(df=df, plot_column='impact', count_column='sla_breach_count', plot_title='SLA Breaches Distribution Across Impact Levels')