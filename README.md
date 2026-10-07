# Grammar scoring from speech (SHL hiring assessment 2026)

Predict a grammar score from 0 to 5 for speech clips of 45 to 60 seconds. There are 769 training clips and 216 test clips, scored by RMSE and Pearson correlation.

## Files

- `final_pipeline.ipynb`: the whole pipeline, with the report in the markdown cells.
- `submit_offline.ipynb`: loads the features and predictions saved by the first notebook and writes the submission. The scored Kaggle notebook had no internet, so it could not download the models.
- `experiments.ipynb`: things that did not help (GPT-2 features, pause features, hand-made features) and a few checks on the data files.
- `requirements.txt`

## What it does

1. **Noise clips.** 37 of the training clips are labelled 0, which the rubric does not define. They are pure noise. A speech clip has pauses (about 44% of it is above a small amplitude floor), while these are around 93%, and one threshold at 0.87 separates all 769 training clips. The margin is thin (0.863 against 0.871), so I checked the test set too: one clip goes over the threshold, and the next one is at 0.806. That clip gets 0, and the models are trained on the 732 speech clips only.
2. **Transcripts.** Whisper small through faster-whisper, English, voice filter on. Saved to csv as it runs.
3. **Features.** Hidden states of every layer of `bert-base-uncased` and `roberta-base`, averaged over the tokens. I scored each layer on its own with ridge regression and used the average of a few middle layers (7 to 10 for BERT, 6 to 10 for RoBERTa), since the last layer was not the best one.
4. **Models.** One ridge regression per encoder (predictions averaged), and `roberta-base` fine-tuned with a regression head (5 folds, 4 epochs, lr 2e-5). The final prediction is a 50/50 average of the two, with 0 for the noise clip.

## Results

5-fold CV on the 732 speech clips. The ridge rows for a single encoder are averaged over 3 repeats of the folds. The averaged ridge models, the fine-tuned model and the blend each come from one split, so compare those with some care. A re-run can move the third decimal.

| model | RMSE | Pearson |
|---|---|---|
| predict the mean | 0.998 | n/a |
| BERT, last layer | 0.669 | 0.751 |
| BERT, layer 8 | 0.650 | 0.768 |
| BERT, mean of layers 7-10 | 0.653 | 0.766 |
| RoBERTa, mean of layers 6-10 | 0.635 | 0.780 |
| BERT and RoBERTa concatenated | 0.625 | 0.788 |
| average of the two ridge models | 0.624 | 0.790 |
| RoBERTa fine-tuned | 0.632 | 0.797 |
| final blend (50/50) | 0.603 | 0.806 |

Training RMSE of the two ridge models averaged: 0.517. The fine-tuned model's training RMSE is printed in section 6 of the notebook.

Leaderboard (216 test clips, RMSE): 0.399 for the ridge ensemble alone, 0.393 for the final blend.

## Notes

- `sample_submission.csv` has 204 rows and most of its filenames are not in the test folder. `test.csv` has 216 rows that match the 216 audio files. SHL confirmed the sample file only shows the format and that `test.csv` is what counts, so the submission has 216 rows.
- Some filenames (audio_0, audio_1, ...) exist in both the train and the test folder. I compared the raw bytes and they are different recordings.
- GPT-2 surprise features and pause / speech-rate features gave no gain on top of the BERT features.

## Limitations

Whisper's own mistakes go into the features. The test set is small, so the leaderboard moves a bit with small changes. The noise rule is a hand-picked threshold. Fine-tuning changes a little from run to run.

## Running it

Run `final_pipeline.ipynb` on a Kaggle GPU (it saves the transcripts and the arrays to the working folder), then `experiments.ipynb` if you want the dead ends. For the offline submission, upload the eight files listed at the top of `submit_offline.ipynb` as a dataset and run that notebook with internet off.
