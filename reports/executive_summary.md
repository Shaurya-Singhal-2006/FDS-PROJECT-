# Executive summary: Customer Lifetime Value Prediction Engine

## Decision

Use customer transaction history to rank accounts by expected revenue over the next six months, then combine that ranking with short-term inactivity risk to prioritize marketing. The output is a targeting aid, not a promise of customer spend.

## Data and method

This analysis uses the real UCI Online Retail II data (1,067,371 transaction lines, Dec 2009–Dec 2011; CC BY 4.0). Cleaning leaves 779,425 positive purchase lines and removes 243,007 rows without customer IDs, 26,479 exact duplicates, 18,390 cancellations and 70 non-positive prices. The 18,744 customer-linked negative-quantity rows all have C-prefixed invoices: they are excluded from purchase modeling and retained separately for a pre-cutoff return-rate feature. At the 1 March 2011 forecast cutoff, features use only earlier transactions and the target is revenue in the next six months. Validation uses a later time snapshot than training, with training target windows ending before validation starts. Historical average, BG/NBD + Gamma-Gamma, and gradient boosting are compared; inactivity classification estimates whether a customer makes no purchase in the first 90 days after cutoff.

## Findings

Five descriptive findings from the cleaned data:

- 72.4% of 5,878 identified customers placed at least two orders.
- The UK accounts for 82.8% of cleaned revenue, consistent with the retailer's home market.
- The ten highest-revenue products contribute 8.6% of sales, so revenue is spread across the product range.
- The top 20% of customers contribute 77.2% of revenue, showing substantial customer concentration.
- Q4 generated 35.5% of revenue, above its 25% share of calendar quarters and consistent with a seasonal peak.

These are descriptive patterns, not causal explanations.

On the March–August 2011 chronological holdout, BG/NBD + Gamma-Gamma achieved £558.32 MAE, £1,661.35 RMSE and 0.782 R². It captured 77.8% of future revenue in its top predicted quintile, compared with 76.5% for the historical-average baseline. Gradient boosting was weaker (0.282 R²). Monetary history was the strongest positive permutation-importance driver in the tree model. Champions comprise 887 customers and 71.5% of predicted portfolio revenue; At-Risk contains 21 customers (2.0%). Full segment sizes and actions are in `segment_actions.csv`.

## Recommended actions

- **Champions:** protect with VIP service and early access; avoid blanket discounts.
- **Loyal:** use relevant replenishment and cross-sell messages.
- **Potential:** test low-cost onboarding and second-purchase nudges.
- **At-Risk:** prioritize personalized win-back outreach where predicted value justifies contact cost.
- **Hibernating:** use a capped, low-cost reactivation test; suppress customers who do not respond.

An illustrative Champions + At-Risk scenario assumes 30% margin, 10% incremental retained revenue and £1 per contact, producing £92,321 estimated net benefit across 908 customers. This is not a measured campaign result; replace all assumptions with experiment data before budgeting.

## Limitations

The data contains transaction behavior only, lacks demographics and marketing exposure, represents a UK retailer, and ends in 2011. Monetary value is recorded as transaction price and is not profit. The time holdout is limited, churn is operationally defined as no purchase in the first 90 days after cutoff, and the ROI scenario is not causal evidence. Predictions require fresh, comparable transaction data for production use.

## Five-slide stakeholder outline

1. **Business question:** where to focus retention spend and the six-month value forecast use case.
2. **Evidence:** dataset scope, cleaning choices and leakage-safe time split.
3. **Performance:** model comparison, revenue captured by the top 20%, and key drivers.
4. **Actions:** segment sizes, revenue concentration and recommended treatments.
5. **Pilot and guardrails:** test design, ROI inputs to measure, limitations and next steps.
