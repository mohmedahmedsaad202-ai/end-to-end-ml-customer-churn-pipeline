import pandas as pd
import numpy as np

def generate_customer_data(samples=1000):
    np.random.seed(42)
    data = {
        'customer_id': range(1000, 1000 + samples),
        'tenure_months': np.random.randint(1, 72, samples),
        'monthly_charges': np.round(np.random.uniform(20.0, 120.0, samples), 2),
        'total_support_calls': np.random.randint(0, 10, samples),
        'contract_type': np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], samples, p=[0.5, 0.3, 0.2]),
        'churn': np.random.choice([0, 1], samples, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    df.to_csv('customer_churn.csv', index=False)
    print(f"Generated {samples} rows of customer data -> 'customer_churn.csv'")

if __name__ == "__main__":
    generate_customer_data()
