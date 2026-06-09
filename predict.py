import pandas as pd
import sys
from predictor import LoanDefaultPredictor

# Argument order:
# 1. application_train.csv  (required)
# 2. bureau.csv
# 3. bureau_balance.csv
# 4. previous_application.csv
# 5. POS_CASH_balance.csv
# 6. installments_payments.csv
# 7. credit_card_balance.csv

def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python predict.py <application.csv> "
            "[bureau.csv] [bureau_balance.csv] [previous_application.csv] "
            "[POS_CASH_balance.csv] [installments_payments.csv] [credit_card_balance.csv]"
        )
        sys.exit(1)

    print("📂 Loading application data...")
    application_df = pd.read_csv(sys.argv[1])
    print(f"   ✓ {len(application_df):,} records loaded")

    # Map positional args → kwargs
    arg_map = {
        2: ('bureau_df',   'bureau.csv'),
        3: ('bb_df',       'bureau_balance.csv'),
        4: ('prev_df',     'previous_application.csv'),
        5: ('pos_df',      'POS_CASH_balance.csv'),
        6: ('inst_df',     'installments_payments.csv'),
        7: ('cc_df',       'credit_card_balance.csv'),
    }

    kwargs = {}
    for idx, (kwarg_name, label) in arg_map.items():
        if len(sys.argv) > idx:
            print(f"📂 Loading {label}...")
            kwargs[kwarg_name] = pd.read_csv(sys.argv[idx])
            print(f"   ✓ {len(kwargs[kwarg_name]):,} rows loaded")

    print("\n🤖 Running predictions...")
    predictor = LoanDefaultPredictor()
    probabilities = predictor.predict_proba(application_df, **kwargs)

    result = pd.DataFrame({
        'SK_ID_CURR':          application_df['SK_ID_CURR'],
        'default_probability': probabilities.round(6),
        'prediction':          (probabilities >= 0.5).astype(int)
    })

    output_file = "predictions.csv"
    result.to_csv(output_file, index=False)

    print(f"\n✅ Predictions saved to {output_file}")
    print(f"   Total: {len(result):,}")
    print(f"   Predicted default (1): {result['prediction'].sum():,}")
    print(f"   Predicted repay  (0): {(result['prediction'] == 0).sum():,}")
    print("\nSample output:")
    print(result.head())

if __name__ == "__main__":
    main()