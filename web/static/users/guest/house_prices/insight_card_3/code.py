from agentpoirot.tools import func_tools
import pandas as pd

# Load the dataframe
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/guest/house_prices/dataset.csv')

# Data Manipulation
# Create a new column to calculate the gap between YearBuilt and YearRemodAdd
df['YearGap'] = df['YearRemodAdd'] - df['YearBuilt']

# Create a column to flag properties where the gap is more than 50 years
df['GapOver50'] = df['YearGap'] > 50

# Plotting
# Use a bar plot to visualize the count of properties with a gap over 50 years
func_tools.plot_bar(df=df, plot_column='GapOver50', count_column='Id', plot_title='Properties with Year Built and Remod Gap Over 50 Years')