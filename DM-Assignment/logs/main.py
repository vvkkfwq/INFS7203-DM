======================================================================
GENERATING FINAL SUBMISSION
======================================================================

[1/6] Loading training data...
  ✓ Loaded 10853 samples with 43 features
  ✓ Class distribution: {0: np.int64(8170), 1: np.int64(2683)}

[2/6] Preprocessing data...

   Doing Imputation: Global Mean / Mode

   Doing outlier detection: LOF_k20_c0.01

   Removed=109 (1.0%)

   Doing feature encoding: OrdinalEncoder


   Doing Imputation: Global Mean / Mode

   Skipping outlier detection for test data

   Doing feature encoding: OrdinalEncoder

  ✓ Preprocessing complete
  ✓ Missing values after preprocessing(train set): 0
  ✓ Missing values after preprocessing(test set): 0

[3/6] Creating models: RF + DT with VotingClassifier...

   Model Configuration:

     Random Forest Parameter:
      - n_estimators: 100
      - max_depth: 10
      - min_samples_split: 8
      - min_samples_leaf: 2
      - max_features: None
      - class_weight: balanced
      - random_state: 42

     Decision Tree Parameter:
      - criterion: gini
      - splitter: random
      - max_depth: 10
      - min_samples_split: 30
      - min_samples_leaf: 2
      - max_features: None
      - class_weight: balanced
      - random_state: 42

      Voting Classifier:
      - voting: hard
      - weights: [1, 1]

[4/6] Training and evaluating...
   Running 5-fold cross-validation...
   Cross-validation completed!
   Training final model on complete dataset...
   Final model training completed!

 Cross-validation results:
  ✓ Accuracy: 0.795 ± 0.009
  ✓ F1 Score: 0.650 ± 0.012

[5/6] Predict test set...

[6/6] Saving submission file...
 Submission file saved: s4860387.infs4203
======================================================================

 Done!
======================================================================
