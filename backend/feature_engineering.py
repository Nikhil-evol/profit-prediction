import pandas as pd
import numpy as np




def disc_bucket(d):
    if d == 0:
        return "0_none"
    elif d <= 0.1:
        return "1_low"
    elif d <= 0.3:
        return "2_medium"
    elif d <= 0.5:
        return "3_high"
    return "4_extreme"



def feature_extraction(df):

    df = df.copy()

    # Estimated monetary value of the discount applied to an order.
    # Helps quantify the financial impact of discounting strategies.
    df['discount_amount'] = df['Sales'] * df['Discount']

    # Ratio of shipping cost relative to order value.
    # Useful for identifying orders where logistics costs are disproportionately high.
    df['shipping_cost_ratio'] = (
                df['Shipping Cost'] / (df['Sales'] + 1)
         )

    # Average revenue generated per unit sold.
    # Helps distinguish high-value products from low-value products.
    df['sales_per_unit'] = (
             df['Sales'] / df['Quantity']
           )



    df["discount_bucket"] = df["Discount"].apply(disc_bucket)

    df["has_discount"] = (df["Discount"] > 0).astype(int)
    df["high_discount"] = (df["Discount"] > 0.3).astype(int)

    df["log_sales"] = np.log1p(df["Sales"])
    df["log_shipping_cost"] = np.log1p(df["Shipping Cost"])

    # Category flags
    df['is_technology']      = (df['Category'] == 'Technology').astype(int)
    df['is_furniture']       = (df['Category'] == 'Furniture').astype(int)
    df['is_office_supplies'] = (df['Category'] == 'Office Supplies').astype(int)

    # Segment flags
    df['is_consumer']    = (df['Segment'] == 'Consumer').astype(int)
    df['is_corporate']   = (df['Segment'] == 'Corporate').astype(int)
    df['is_home_office'] = (df['Segment'] == 'Home Office').astype(int)


    return df