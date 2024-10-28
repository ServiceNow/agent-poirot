from agentpoirot.tools import func_tools
import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv('/mnt/home/projects/research-skilled-poirot/web/static/users/issam.laradji@servicenow.com/customer_service_management/dataset.csv')

# Data Manipulation
# Create a new column to categorize case resolution times based on priority
df['priority_case_duration'] = df['rpt_case_resolve_duration']  # Use resolution duration for analysis
# Create a new column to identify high and low priority cases
df['priority_category'] = df['priority'].apply(lambda x: 'High' if x == 'High' else 'Low')  # Categorize based on priority

# Stage Name: Plotting
# Use the plot_boxplot function to visualize case resolution times for high-priority vs low-priority cases
func_tools.plot_boxplot(df=df, x_column='priority_category', y_column='priority_case_duration', plot_title='Case Resolution Times for High vs Low Priority Cases')