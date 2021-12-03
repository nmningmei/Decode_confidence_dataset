Contributors: 
Maxine T. Sherman <m.sherman@sussex.ac.uk>
Warrick Roseboom <w.roseboom@sussex.ac.uk>
Anil K. Seth <a.k.seth@sussex.ac.uk>

Citation:
None (manuscript in preparation)

Stimulus:
We used the Positive Evidence stimuli from Koizumi et al (2015). For a full explanation, please see their paper.
Briefly, the stimulus comprised three components:
(i) One left-oriented Gabor
(ii) One right-oriented Gabor
(iii) Visual white noise

On each trial, either (i) or (ii) was higher.
The task was to report the orientation of the higher contrast Gabor.

Stimulus coding:
0 = right Gabor has higher contrast
1 = left Gabor has higher contrast

Confidence scale:
Confidence rating from 50 to 100

Manipulations:
This was a 2 x 2 within-subjects design.
- Factor 1: Positive evidence (low, high). 
            For low positive evidence stimuli, the contrast of the correct orientation was relatively low.
            We set contrast_incorrect/contrast_correct to 0.7 and used QUEST to titrate contrast_correct/contrast_noise to achieve 72% correct.
            For high positive evidence stimuli, the contrast of the correct orientation was relatively high.
            We set contrast_incorrect/contrast_correct to 0.3 and used QUEST to titrate contrast_correct/contrast_noise to achieve 72% correct.
            These factor levels respectively drive low and high decision confidence, while leaving decision accuracy similar.

- Factor 2: Expected positive evidence (low, high). 
            This was manipulated block-wise.
            In 'Low' blocks, 75% of trials had low positive evidence stimuli and 25% of trials had high positive evidence stimuli.
            In 'High' blocks, 75% of trials had high positive evidence stimuli and 25% of trials had low positive evidence stimuli.
            Because low/high positive evidence stimuli drive low/high confidence, the Low/High blocks should cause subjects to have low/high self-efficacy.

Condition coding:
 - 1: Low positive evidence stimulus, Expect low positive evidence
 - 2: Low positive evidence stimulus, Expect high positive evidence
 - 3: High positive evidence stimulus, Expect low positive evidence
 - 4: High positive evidence stimulus, Expect high positive evidence 

Block Size:
80 trials per block

Feedback:
None

Subject population:
46 healthy subjects (age = 19.8 +/ 1.1, 39 female)

Response device:
Keyboard

Experiment setting:
Lab

Training:
Approximately 10 minutes

Experiment goal:
Test whether decision confidence is biased by priors on confidence (self-efficacy)

Main result:
Confidence is more sensitive to experienced difficulty (positive evidence) under high self-efficacy

Special instructions:
Subjects were told that in each block, either the high or low positive evidence stimulus would be more likely. However, the expectation had to be learnt implicitly over the course of the block.

Link to material/codes:
n/a

Experiment dates:
2018-2019

Location of data collection:
University of Sussex, UK

Any other important information about the dataset:
On each trial, the stimulus is built by combining the three components: 
 - the 'positive', i.e. correct gabor
 - the 'negative', i.e. incorrect gabor
 - the noise
Each has a different contrast, and these contrasts differ as a function of low vs. high positive evidence condition.
The contrasts for each of these 3 components in the 2 positive evidence conditions are given in the dataset.
Reaction times for confidence were not collected.

A more thorough outline of the experiment can be found in our OSF pre-registration https://osf.io/yhv2k