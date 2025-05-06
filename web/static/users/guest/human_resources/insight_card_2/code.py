from agentpoirot.tools import func_tools
import pandas as pd

# Load the dataframe
df = pd.read_csv('/Users/issam.laradji/projects/agent-poirot/web/static/users/guest/human_resources/dataset.csv')

# Data Manipulation
# Create a new column to indicate if an employee was promoted in the last 2 years and left the company
df['promoted_and_left'] = (df['promotion_last_2_years'] == 1) & (df['left_company'] == 1)

# Plotting
# Use a bar plot to visualize the relationship between recent promotions and employee turnover
func_tools.plot_bar(df=df, plot_column='promoted_and_left', count_column='employee_id', plot_title='Relationship between Recent Promotions and Employee Turnover')