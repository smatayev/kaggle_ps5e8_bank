import pandas as pd

def blend_submissions():
    """
    Blends the predictions from the weighted average and stacking models.
    """
    print("Blending submission files...")

    # Load the two submission files
    df_weighted = pd.read_csv("submission_simple_avg.csv")
    df_stacking = pd.read_csv("submission_stacking.csv")

    # Define the blend weights (a 50/50 split is a great starting point)
    weight_weighted = 0.3
    weight_stacking = 0.7

    # Create the final blended prediction
    df_blend = df_weighted.copy()
    df_blend['y'] = (weight_weighted * df_weighted['y']) + \
                    (weight_stacking * df_stacking['y'])

    # Save the final submission file
    final_submission_path = "submission_final_blend.csv"
    df_blend.to_csv(final_submission_path, index=False)

    print(f"Final blended submission created at '{final_submission_path}'")
    print(df_blend.head())

if __name__ == "__main__":
    blend_submissions()
