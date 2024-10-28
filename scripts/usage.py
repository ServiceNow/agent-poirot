import pandas as pd
import agentpoirot


# Load Data
sales_by_country = pd.DataFrame(
    {
        "State": ["California", "Texas", "Florida", "New York", "Alaska"],
        "Sales": [50000, 45000, 40000, 42000, 100000],
        "Customers": [
            500,
            450,
            400,
            420,
            50,
        ],
    }
)

# set your openai api key
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Get Agent
agent = agentpoirot.Poirot(
    table=sales_by_country,
    model_name="gpt-4o-mini",
    meta_dict={
        "goal": "find surprising insights in terms of how factors are correlated",
    },
)

# Generate Insights
insights = agent.generate_insights(n_insights=1, as_str=True)
print(insights)
