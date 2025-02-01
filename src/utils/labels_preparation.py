import pandas as pd


def process_labels(input):

  df_label = pd.read_csv(input)

  mean_value = df_label['Average cohesion score'].mean()
  df_label.loc[:, 'Average cohesion score'] = df_label['Average cohesion score'].fillna(mean_value)

  # Extract the column of interest (as there's multiple column in the file)
  cohesion_score = df_label["Average cohesion score"].values

  if len(cohesion_score) % 2 != 0:
   raise ValueError("Check the dataset, something wrong with rows - they are not even.")

  # Reshape it into pairs
  cohesion_pair = cohesion_score.reshape(-1, 2)

  pair_mean = cohesion_pair.mean(axis=1)

  # Set arbitrary threshold for cohesion score
  # 1 = High Cohesive, 0 = Low Cohesive
  cohesion_binary_set = (pair_mean > 4.5).astype(int)

  # Create a new DataFrame
  cohesion_binary = pd.DataFrame({
  'Labels': cohesion_binary_set
  })

  return cohesion_binary