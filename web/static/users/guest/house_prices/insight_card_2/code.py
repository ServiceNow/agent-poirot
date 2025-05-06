# Import necessary libraries
import pandas as pd
from agentpoirot.tools import func_tools

# Load the dataframe
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/guest/house_prices/dataset.csv')

# Stage: Data Manipulation
# Calculate the average sale price per neighborhood
df['NeighborhoodAvgPrice'] = df.groupby('Neighborhood')['SalePrice'].transform('mean')

# Calculate the percentage difference from the neighborhood average
df['PriceDiffPercent'] = (df['SalePrice'] - df['NeighborhoodAvgPrice']) / df['NeighborhoodAvgPrice'] * 100

# Create a column to flag properties where the price is more than 30% above or below the neighborhood average
df['PriceAnomaly'] = df['PriceDiffPercent'].abs() > 30

# Stage: Plotting
# Plot the bar chart to visualize the count of properties with price anomalies per neighborhood
func_tools.plot_bar(df=df, plot_column='Neighborhood', count_column='PriceAnomaly', plot_title='Properties with Price Anomalies by Neighborhood')