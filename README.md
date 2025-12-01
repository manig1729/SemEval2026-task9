# Structure

The goal of this project is to accurately classify polarization in tweets. We are given a gold standard training set with polarization classes.

Methods:
- Run pointwise mutual information between polarized and non-polarized tweets to discover high-frequency words.
- Use these words to create a lexicon of spurious correlations. Mask them with probability p to ensure the model does not just learn the identity word.
- Evaluate this on the original dev set and also the masked dev set.
- Try upweighting non-polarized texts with identity words.

- Also try counterfactual augmentation where we swap the words from the lexicon with other words.

- Try using an LLM to rephrase some of the tweets and classify them as polarized/non-polarized.

- Try a multitask model for all three subtasks.
- Find optimal thresholds for each label in 2 and 3.