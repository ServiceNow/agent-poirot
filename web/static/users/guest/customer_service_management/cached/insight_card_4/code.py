from agentpoirot.tools import func_tools
import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv(
    "/mnt/home/projects/research-skilled-poirot/web/static/users/issam.laradji@servicenow.com/customer_service_management/dataset.csv"
)

# Stage 1: Data Manipulation
# Create a new column to calculate the time taken to resolve each ticket
df["resolution_time"] = (
    pd.to_datetime(df["resolved_at"]) - pd.to_datetime(df["opened_at"])
).dt.total_seconds() / 3600  # Convert to hours

# Create a new column to flag cases resolved within the first 24 hours
df["resolved_within_24_hours"] = (
    df["resolution_time"] <= 24
)  # Boolean flag for resolution within 24 hours

# Stage 2: Plotting
# Use the count of cases resolved within 24 hours for the plot
func_tools.plot_bar(
    df=df,
    plot_column="resolved_within_24_hours",
    count_column="resolution_time",
    plot_title="Customer Support cases Resolved Within 24 Hours",
)
